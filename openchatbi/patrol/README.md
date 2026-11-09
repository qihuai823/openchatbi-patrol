# Patrol — 把被动问答 agent 改成主动巡检

## 为什么需要它

OpenChatBI 的主 agent 是**被动**的：用户提问，它调工具，它回答。整张图的输入是 `MessagesState`，没有 `HumanMessage` 就没有任何事发生。

真实的业务监控不是这样运作的。没有人会每天早上九点准时问「昨天 GMV 正常吗」。需要的是：**每天定时扫一遍，只有出问题时才来找人。**

这两种形态的差别不是"加个定时器"，而是：

| | 被动问答 | 主动巡检 |
|---|---|---|
| 触发 | 用户消息 | 调度器 |
| 每次成本 | 一次 LLM 调用 | 0（纯统计 + SQL） |
| 输出 | 自然语言 | 结构化告警 |
| 核心难点 | 理解问题 | **不误报、不重复报** |
| 失败模式 | 答错 | 噪音淹没真问题 |

表格最后一行是关键：**巡检系统最大的失败不是漏报，是误报**。一天推 20 条垃圾告警的巡检，等于没有巡检。所以这个模块一半的代码在处理"什么时候**不**推"。

## 为什么不做成 agent graph 的一个节点

考虑过。结论是不该。

- 图里每个节点消费和生产 `messages`，巡检没有对话
- 图靠 LLM 决定下一步，巡检的判定必须是**确定性**的（同一份数据跑两次必须得出同一结论，否则没法做回归测试）
- 巡检要能在 cron 上跑，不能因为模型服务抖动就整个失败

所以它是一条独立的同步管线，**不依赖 LLM、不依赖 Docker、不依赖预测服务**。两者在人和 agent 交互的地方汇合：告警里的 `metric / bucket / score / direction` 正好是 agent `text2sql` 工具需要的下钻上下文。

## 架构

```
plan.py      声明式计划：看哪些指标、阈值多少、哪些已知问题要抑制
   │
baseline.py  统计基线：中位数 + MAD（退化时退 IQR），周期性感知
   │
dedup.py     四道闸：严重度下限 → 冷却 → 升级覆盖 → 提醒
   │
store.py     SQLite：告警状态（冷却记忆）+ 决策审计日志
   │
runner.py    编排：取数 → 判定 → 过闸 → 推送 → 标记恢复
   │
notify.py    Console / JSONL / Webhook
```

每个模块都可以单独测。`baseline` / `dedup` / `plan` 是纯函数，`store` 和 `runner` 用临时 SQLite 测。

## 四道去重闸

按顺序执行，任何一道拦下就不推：

1. **严重度下限** —— 低于 `min_severity` 的偏差只记录不推送。日抖动不该叫醒人。
2. **冷却窗口** —— 同一个指纹在 `cooldown_hours` 内不重复推。这是"持续性问题只响一次"的实现。
3. **升级覆盖** —— 同一指纹严重度**等级**上升时，立刻推送，不等冷却。问题恶化不能等定时器。
   注意是等级上升（medium→critical），不是分数上升（0.85→0.95 都是 critical，不算升级，否则会因抖动反复推）。
4. **提醒** —— 冷却到期且问题仍未恢复，再推一次。否则长期问题会静默消失。

外加独立的**抑制规则**：已知非问题（数据不完整的时段、计划内迁移窗口）按 glob 匹配直接拦掉，但必须写明 `reason`——**没有理由的抑制等同于 bug**，所以 plan 校验会强制要求。

## 告警指纹：为什么不含时间桶

```python
AlertFingerprint(metric="gmv", dimension="category", dimension_value="electronics", direction=None)
# key = "gmv|category=electronics"
```

指纹里**故意不放日期**。同一个类目连续三天异常，是**同一个**还在持续的问题，不是三个新问题。日期放在 `Alert.deviation.bucket` 里用于展示，不参与去重。

如果指纹带日期，每天都会算"首次出现"，冷却机制直接失效。

## 为什么不用 `analysis/anomaly_detection.py`

那个模块更精密（预测残差 + 方向权重 + 流量调制 + 持续时间加成），但它**必须调外部预测服务**：

```python
if not check_forecast_service_health():
    return 0.0, {"error": "Time Series Forecasting Service Unavailable..."}
```

也就是说巡检要依赖一个跑在 Docker 里、端口 8765 的模型服务。这在 cron 环境里是灾难——服务挂了巡检就哑了，而且哑了没人知道。

巡检用中位数 + MAD：

- **鲁棒**：MAD 有 50% 崩溃点，一两天的极端值污染不了基线
- **零依赖**：纯 Python，任何环境都能跑
- **可解释**：`z = 3.2，基线 1122，实际 227` 比"模型残差 0.86"好解释得多

代价是它对复杂季节性的建模不如预测模型。对日粒度的经营指标够用，对分钟级流量指标不够——那是另一个取舍。

## 踩到的坑：不完整的尾部时间桶

**第一版跑真实数据，19 条序列里 17 条报警。** GMV −12.6%，订单量 −20.7%，大半类目暴跌。

不是业务出问题，是 `2026-05-31` 当天数据只到 21:32——**这一天还没过完**。巡检拿半天数据跟完整天比，当然全线下跌。

这正是监控系统最经典的误报来源。修法（`runner._drop_incomplete_tail`）：

```python
last_date = parse_bucket(series[-1].bucket)
if last_date is not None and last_date >= run_date:
    return series[:-1]   # 当前周期还没结束，丢掉
```

只丢**尾部且日期 ≥ 运行日**的桶，历史桶一律不动——所以回填、重跑关闭周期都不受影响。用 `>=` 而不是 `==`，是为了顺便挡住未来日期的脏数据。

效果对比（同一份数据，不同运行时刻）：

| 运行时刻 | 判定的最新桶 | 推送数 |
|---|---|---|
| 2026-05-31 21:00（当天未结束） | 2026-05-30 | 1（office 真实下跌） |
| 2026-06-01 09:00（5-31 已结束） | 2026-05-31 | 2（beauty、outdoor 真实上涨） |

对应测试：`tests/patrol/test_runner.py::test_incomplete_trailing_bucket_is_ignored`。

## 用法

```bash
# 预演：不推送、不写状态
python -m openchatbi.patrol --plan example_ecommerce/patrol_plan.yaml --dry-run

# 回放某个时刻（调阈值时用）
python -m openchatbi.patrol --plan example_ecommerce/patrol_plan.yaml \
    --now 2026-06-01T09:00 --dry-run

# 真实定时运行
python -m openchatbi.patrol --plan patrol_plan.yaml \
    --state /var/lib/patrol/state.sqlite \
    --notify jsonl --jsonl /var/log/patrol/alerts.jsonl

# 查审计：某条为什么没推
python -m openchatbi.patrol --plan patrol_plan.yaml --state ... --history 20
```

退出码区分三种状态，方便调度器判断：`0` 正常 / `1` 计划或数据源不可用 / `2` 跑完了但部分指标出错。

## 计划文件

```yaml
version: 1
database: example_ecommerce/ecommerce.sqlite

defaults:
  history_points: 35
  min_samples: 14
  min_relative_change: 0.08   # 小于基线的 8% 视为噪音
  cooldown_hours: 24
  min_severity: medium
  escalate: true
  weekday_aware: true
  drop_incomplete_tail: true

metrics:
  - name: gmv
    label: 成交总额 GMV
    direction: both            # both | rise | drop
    sql: |
      SELECT date(o.date_order_placed) AS bucket, SUM(oi.line_amount) AS value
      FROM Orders o JOIN Order_Items oi ON oi.order_id = o.order_id
      WHERE o.order_status != 'Canceled'
      GROUP BY 1 ORDER BY 1

  - name: gmv_by_category
    dimension: category        # 声明维度后，SQL 必须多返回一列，每个取值一条序列
    direction: both
    sql: |
      SELECT date(o.date_order_placed) AS bucket, p.category AS category, SUM(oi.line_amount) AS value
      ...

suppress:
  - reason: "2025-11 只含 11-30 单日数据，是数据不完整而非业务下滑"
    bucket_glob: "2025-11*"
```

SQL 只要求返回 `bucket` 和 `value` 两列（声明了 `dimension` 就多一列）。加一个监控指标 = 改配置，不改代码。

## 已知限制

- **只评估最新一个桶**。不回填历史告警，所以不能用来补发过去漏掉的告警。
- **日/月粒度**。`weekday_aware` 只对日粒度有意义；月粒度会退化成整体基线。
- **无季节性建模**。春节、大促这类结构性变化会被当成异常——需要靠 `suppress` 规则人工压制，或者后续引入同比基线。
- **单机 SQLite**。多实例并发跑会锁冲突，需要换成 PG 或加文件锁。
- **未接入主图**。目前是独立 CLI。接入方式是在 agent 侧加一个 `patrol_status` 工具，让人可以问"昨天有哪些告警"，再由 `text2sql` 下钻。

## 测试

```bash
pytest tests/patrol -o addopts=""
# 65 passed
```

覆盖：基线鲁棒性（含单点巨值污染）、星期周期性、评分曲线断点、四道闸的每条分支、抑制规则匹配、计划校验的全部失败路径、端到端运行、冷却抑制、dry-run 不落状态、维度拆分、恢复标记、不完整尾部桶。
