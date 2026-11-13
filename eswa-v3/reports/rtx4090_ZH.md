# RTX 4090 独立样本验证：中文结果报告

结果生成时间：2026-11-13T06:43:12.480646+00:00。本轮只报告实验，不修改论文。

## 完成状态与冻结

协议、输入/代码/权重及校准 ε 已按顺序冻结；两模型、全部配置、两次重复和跨硬件单次均已完成。校准先于新评估题打分。主分析沿用旧 ε，次要分析单列重新校准 ε，未根据评估结果调参。

## 样本与来源

实际样本：{"evaluation/qasper/dev": 87, "evaluation/qasper/test": 105, "calibration/qasper/test": 32, "evaluation/hotpot/dev": 200, "calibration/hotpot/dev": 32}。历史排除 6088 个题 ID、448 篇论文；与历史题 ID、QASPER 论文及规范化问题文本重叠均为 0。QASPER 一篇一题；本轮不同论文的问题表述重复计数为 1，题 ID 和论文均不同，未事后替换样本。sample_manifest.csv 保留全部问题表述及来源。HotpotQA 为官方 dev 内容，HF 固定版本传输，不声称 CMU JSON 字节身份。

## 校准 ε（每模型一个）

两个数据集各 32 题、batch4/8/32、两次重复及全部候选共同取最大绝对误差，再乘 1.5。

|模型|沿用 ε|校准最大误差|新 ε|
|---|---:|---:|---:|
|qwen|0.9375|0.5|0.75|
|bge|0.1875|0.21875|0.328125|

## 全部主分析和次要分析结果

下表质量为 rep0；CSV 保留 rep0/rep1 全部指标。耗时为每题两次中位数，表中耗时比为平均耗时之比；逐题比例的均值/中位数及超时题数也在 CSV。重复之间如有质量差异，紧随表格列出。

|模型/数据|方法/batch|context差异|重打分均值|gold召回|答案F1|耗时比|慢于serial题数|
|---|---|---:|---:|---:|---:|---:|---:|
|bge/hotpot|band_recalibrated/4|0/200|0.089932|0.792762|0.526416|0.3705|0|
|bge/hotpot|band_recalibrated/8|0/200|0.089989|0.792762|0.526416|0.2486|0|
|bge/hotpot|band_recalibrated/32|0/200|0.089908|0.792762|0.526416|0.1623|0|
|bge/hotpot|band_transferred/4|0/200|0.074377|0.792762|0.526416|0.3576|0|
|bge/hotpot|band_transferred/8|0/200|0.074647|0.792762|0.526416|0.2341|0|
|bge/hotpot|band_transferred/32|0/200|0.074519|0.792762|0.526416|0.1491|0|
|bge/hotpot|gate_v2/4|0/200|0.007029|0.792762|0.526416|0.3021|0|
|bge/hotpot|gate_v2/8|0/200|0.006883|0.792762|0.526416|0.1777|0|
|bge/hotpot|gate_v2/32|0/200|0.006955|0.792762|0.526416|0.0906|0|
|bge/hotpot|native/4|15/200|0.000000|0.794905|0.529750|0.2830|0|
|bge/hotpot|native/8|17/200|0.000000|0.792762|0.524448|0.1603|0|
|bge/hotpot|native/32|19/200|0.000000|0.792762|0.524448|0.0745|0|
|bge/hotpot|serial/1|0/200|0.000000|0.792762|0.526416|1.0000|0|
|bge/hotpot|top15/4|0/200|0.086766|0.792762|0.526416|0.3690|0|
|bge/hotpot|top15/8|0/200|0.086766|0.792762|0.526416|0.2442|0|
|bge/hotpot|top15/32|0/200|0.086766|0.792762|0.526416|0.1593|0|
|bge/qasper|band_recalibrated/4|0/192|0.360556|0.663483|0.468485|0.5974|10|
|bge/qasper|band_recalibrated/8|0/192|0.360907|0.663483|0.468485|0.4807|6|
|bge/qasper|band_recalibrated/32|0/192|0.360603|0.663483|0.468485|0.4090|4|
|bge/qasper|band_transferred/4|0/192|0.321478|0.663483|0.468485|0.5618|7|
|bge/qasper|band_transferred/8|0/192|0.322695|0.663483|0.468485|0.4432|4|
|bge/qasper|band_transferred/32|0/192|0.322196|0.663483|0.468485|0.3741|3|
|bge/qasper|gate_v2/4|0/192|0.035573|0.663483|0.468485|0.3758|1|
|bge/qasper|gate_v2/8|0/192|0.035581|0.663483|0.468485|0.2562|1|
|bge/qasper|gate_v2/32|0/192|0.036760|0.663483|0.468485|0.1870|1|
|bge/qasper|native/4|16/192|0.000000|0.663483|0.464687|0.3059|0|
|bge/qasper|native/8|12/192|0.000000|0.663483|0.466461|0.1884|0|
|bge/qasper|native/32|14/192|0.000000|0.663483|0.465051|0.1184|0|
|bge/qasper|serial/1|0/192|0.000000|0.663483|0.468485|1.0000|0|
|bge/qasper|top15/4|0/192|0.395341|0.663483|0.468485|0.6189|9|
|bge/qasper|top15/8|0/192|0.395341|0.663483|0.468485|0.5011|5|
|bge/qasper|top15/32|0/192|0.395341|0.663483|0.468485|0.4328|4|
|qwen/hotpot|band_recalibrated/4|0/200|0.126025|0.765738|0.474047|0.4089|0|
|qwen/hotpot|band_recalibrated/8|0/200|0.125109|0.765738|0.474047|0.2752|0|
|qwen/hotpot|band_recalibrated/32|0/200|0.125396|0.765738|0.474047|0.2154|0|
|qwen/hotpot|band_transferred/4|0/200|0.151871|0.765738|0.474047|0.4345|0|
|qwen/hotpot|band_transferred/8|0/200|0.151160|0.765738|0.474047|0.3013|0|
|qwen/hotpot|band_transferred/32|0/200|0.151677|0.765738|0.474047|0.2410|0|
|qwen/hotpot|gate_v2/4|3/200|0.013971|0.765738|0.474047|0.3061|0|
|qwen/hotpot|gate_v2/8|5/200|0.014260|0.765738|0.477380|0.1734|0|
|qwen/hotpot|gate_v2/32|4/200|0.014412|0.765738|0.474047|0.1130|0|
|qwen/hotpot|native/4|60/200|0.000000|0.765738|0.481596|0.2842|0|
|qwen/hotpot|native/8|56/200|0.000000|0.768238|0.486547|0.1516|0|
|qwen/hotpot|native/32|51/200|0.000000|0.768238|0.486478|0.0908|0|
|qwen/hotpot|serial/1|0/200|0.000000|0.765738|0.474047|1.0000|0|
|qwen/hotpot|top15/4|0/200|0.086766|0.765738|0.474047|0.3695|0|
|qwen/hotpot|top15/8|0/200|0.086766|0.765738|0.474047|0.2371|0|
|qwen/hotpot|top15/32|0/200|0.086766|0.765738|0.474047|0.1762|0|
|qwen/qasper|band_recalibrated/4|0/192|0.487210|0.667713|0.447888|0.7204|35|
|qwen/qasper|band_recalibrated/8|0/192|0.485859|0.667713|0.447888|0.5895|17|
|qwen/qasper|band_recalibrated/32|0/192|0.486514|0.667713|0.447888|0.5532|17|
|qwen/qasper|band_transferred/4|0/192|0.541561|0.667713|0.447888|0.7712|54|
|qwen/qasper|band_transferred/8|0/192|0.541866|0.667713|0.447888|0.6413|28|
|qwen/qasper|band_transferred/32|0/192|0.541838|0.667713|0.447888|0.6037|26|
|qwen/qasper|gate_v2/4|0/192|0.073259|0.667713|0.447888|0.4034|0|
|qwen/qasper|gate_v2/8|1/192|0.072982|0.667713|0.447864|0.2745|0|
|qwen/qasper|gate_v2/32|0/192|0.071700|0.667713|0.447888|0.2346|0|
|qwen/qasper|native/4|47/192|0.000000|0.673269|0.452658|0.3029|0|
|qwen/qasper|native/8|47/192|0.000000|0.675278|0.445993|0.1732|0|
|qwen/qasper|native/32|54/192|0.000000|0.667316|0.444929|0.1350|0|
|qwen/qasper|serial/1|0/192|0.000000|0.667713|0.447888|1.0000|0|
|qwen/qasper|top15/4|0/192|0.395341|0.667713|0.447888|0.6248|12|
|qwen/qasper|top15/8|0/192|0.395341|0.667713|0.447888|0.4951|8|
|qwen/qasper|top15/32|0/192|0.395341|0.667713|0.447888|0.4568|6|

重复间 context 不一致配置：[]。所有与 serial 的 gold/F1 增减均保留，不预设准确率相同。

## Band 的误差界检查

|模型/数据|方法/batch/rep|ε|最大误差|越界候选|越界题|
|---|---|---:|---:|---:|---:|
|bge/hotpot|band_recalibrated/4/0|0.328125|0.3125|0|0|
|bge/hotpot|band_recalibrated/4/1|0.328125|0.3125|0|0|
|bge/hotpot|band_recalibrated/8/0|0.328125|0.375|1|1|
|bge/hotpot|band_recalibrated/8/1|0.328125|0.375|1|1|
|bge/hotpot|band_recalibrated/32/0|0.328125|0.3408203125|1|1|
|bge/hotpot|band_recalibrated/32/1|0.328125|0.3408203125|1|1|
|bge/hotpot|band_transferred/4/0|0.1875|0.3125|9|8|
|bge/hotpot|band_transferred/4/1|0.1875|0.3125|9|8|
|bge/hotpot|band_transferred/8/0|0.1875|0.375|8|7|
|bge/hotpot|band_transferred/8/1|0.1875|0.375|8|7|
|bge/hotpot|band_transferred/32/0|0.1875|0.3408203125|9|9|
|bge/hotpot|band_transferred/32/1|0.1875|0.3408203125|9|9|
|bge/qasper|band_recalibrated/4/0|0.328125|0.125|0|0|
|bge/qasper|band_recalibrated/4/1|0.328125|0.125|0|0|
|bge/qasper|band_recalibrated/8/0|0.328125|0.1875|0|0|
|bge/qasper|band_recalibrated/8/1|0.328125|0.1875|0|0|
|bge/qasper|band_recalibrated/32/0|0.328125|0.140625|0|0|
|bge/qasper|band_recalibrated/32/1|0.328125|0.140625|0|0|
|bge/qasper|band_transferred/4/0|0.1875|0.125|0|0|
|bge/qasper|band_transferred/4/1|0.1875|0.125|0|0|
|bge/qasper|band_transferred/8/0|0.1875|0.1875|0|0|
|bge/qasper|band_transferred/8/1|0.1875|0.1875|0|0|
|bge/qasper|band_transferred/32/0|0.1875|0.140625|0|0|
|bge/qasper|band_transferred/32/1|0.1875|0.140625|0|0|
|qwen/hotpot|band_recalibrated/4/0|0.75|0.71875|0|0|
|qwen/hotpot|band_recalibrated/4/1|0.75|0.71875|0|0|
|qwen/hotpot|band_recalibrated/8/0|0.75|0.625|0|0|
|qwen/hotpot|band_recalibrated/8/1|0.75|0.625|0|0|
|qwen/hotpot|band_recalibrated/32/0|0.75|0.6875|0|0|
|qwen/hotpot|band_recalibrated/32/1|0.75|0.6875|0|0|
|qwen/hotpot|band_transferred/4/0|0.9375|0.71875|0|0|
|qwen/hotpot|band_transferred/4/1|0.9375|0.71875|0|0|
|qwen/hotpot|band_transferred/8/0|0.9375|0.625|0|0|
|qwen/hotpot|band_transferred/8/1|0.9375|0.625|0|0|
|qwen/hotpot|band_transferred/32/0|0.9375|0.6875|0|0|
|qwen/hotpot|band_transferred/32/1|0.9375|0.6875|0|0|
|qwen/qasper|band_recalibrated/4/0|0.75|0.390625|0|0|
|qwen/qasper|band_recalibrated/4/1|0.75|0.390625|0|0|
|qwen/qasper|band_recalibrated/8/0|0.75|0.421875|0|0|
|qwen/qasper|band_recalibrated/8/1|0.75|0.421875|0|0|
|qwen/qasper|band_recalibrated/32/0|0.75|0.375|0|0|
|qwen/qasper|band_recalibrated/32/1|0.75|0.375|0|0|
|qwen/qasper|band_transferred/4/0|0.9375|0.390625|0|0|
|qwen/qasper|band_transferred/4/1|0.9375|0.390625|0|0|
|qwen/qasper|band_transferred/8/0|0.9375|0.421875|0|0|
|qwen/qasper|band_transferred/8/1|0.9375|0.421875|0|0|
|qwen/qasper|band_transferred/32/0|0.9375|0.375|0|0|
|qwen/qasper|band_transferred/32/1|0.9375|0.375|0|0|

越界严格使用 >ε，等于边界另外计数。经验 ε 的迁移失效不因 context 碰巧一致而隐藏；只有实际误差满足界且真实重打分与参考一致时，band 的条件保证适用。

## 同题跨硬件 serial

|模型/数据|题数|相对3090Ti rep0差异|相对3090Ti rep1差异|
|---|---:|---:|---:|
|qwen/hotpot|200|71|71|
|qwen/qasper|192|45|45|
|bge/hotpot|200|17|17|
|bge/qasper|192|14|14|

此组使用原阶段2完全相同的392道输入，不能当作新题泛化样本。所有编码哈希已核对；逐题差异、旧serial重复差异和分数差异详见CSV。

## GPU占用、无效尝试与局限

GPU 监控摘要（公开版仅列进程数）：{"sample_interval_seconds": 0.5, "state_change_records": 56303, "foreign_process_count_including_queue": 103, "speech_service_process_count_including_queue": 0, "foreign_process_count_during_requests": 0, "contaminated_attempts": 0, "failed_attempts": 0, "limitation": "No detection cannot exclude jobs shorter than the 0.5-second sampling interval."}

独立重放 26896 条请求；核对 249138 个真实重打分值，其中与对应 serial 不同 0 个。错误逐条导出，不以删除样本掩盖。

Reader：{"state": "complete", "utc": "2026-11-13T06:07:40.403415+00:00", "unique_prompts": 1051, "links": 25088, "reused_3090_answers": 0, "repeated_prompts": 16, "repeat_identity": 16, "answers_sha256": "bc3b332cbc947c507dd70d69eaf2a58a93a5477be50af36e085144d105b21ed0", "finished_utc": "2026-11-13T06:39:39.494873+00:00", "seconds": 1919.091498313006}。未复用3090Ti答案。

零差异不是总体零风险；CSV 的单侧95%样本上界为 1−0.05^(1/n)，n不乘重复次数，使用独立同分布近似。局限仍为两种GPU环境、一个reader、一种打包、两个数据集；硬件比较同时受记录的软件/驱动环境差异约束。

阶段1仍记录为未达到预设90%（实际86.49%），继续研究由用户决定。本轮无事后修改阈值、方法或论文。
