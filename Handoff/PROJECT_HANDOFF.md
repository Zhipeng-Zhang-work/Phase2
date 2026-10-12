# LMCache 实验项目详细交接

更新：2026-09-19。本文结合长期对话和本地代码核对整理。历史实测不表示当前所有版本、全部数据集都通过；本轮未连接超算执行实验。

## 1. 目标、边界和用户偏好

研究 RAG 的 KV cache 预计算、复用与选择性重计算，比较答案质量、TTFT、吞吐。主实验为 CacheBlend 多版本及 vLLM full recompute；附加实验为 CacheTune、SpecCache，C2KV 正在准备训练与集成。

用户明确要求：

- 尽量新建独立文件/目录，不修改稳定主实验或其他子实验。
- 看实际代码和调用链，不只看注释，不把猜测说成已证实原因。
- 本地与超算数据路径不同，禁止为适配本地擅自修改 workload 路径。
- 已有 Jupyter GPU allocation 时直接 bash 运行脚本，不要用 sbatch 再提交任务；停止任务也要区分 shell 子进程与整个 allocation。
- 实验禁用 Pyramid。用户手动运行的 YAML 不要顺手修改；批处理使用生成的配置。
- CacheTune nested_union 比例是刻意设定，不要自动调整。
- 原 SpecCache multi 脚本正常工作时，只修对应新变体 wrapper，不要破坏原流程。
- 可以重复跑已完成的数据集；不要未经要求增加跳过逻辑。
- 长上下文已经导致多次额度中断，需保存进度，不重复开展已完成工作。

## 2. 环境和目录

| 项目 | 已知信息 |
| --- | --- |
| 旧本地仓库 | `/Users/zhangzhipeng/Documents/Codes/LMCache` |
| 超算仓库 | `/home/chzhan.chzhipen/LMCache` |
| 用户名 | `chzhan.chzhipen` |
| 超算 Python | 仓库 `.venv`、uv；历史 Python 3.12 |
| vLLM 实测 | 0.10.0，不是早先猜测的0.9.0 |
| torch 历史输出 | 2.7.1+cu126 |
| GPU | H200，以及 A100-SXM4-80GB |
| 旧本地环境 | `/opt/anaconda3/envs/llm/bin/python` |

用户曾叫设备“V100 80GB”，实际打印是 A100；以设备输出为准。HF 外置SSD路径与失效python alias已由用户自行解决，不再排查或改库文件。

Slurm 会执行 `/var/spool/slurm/...` 下的脚本副本，不能无条件由 BASH_SOURCE 定位仓库。相对 SBATCH 输出路径依赖提交目录，日志目录须预先存在。

本次目录核对未见 `.git`，不要假设可以用 git 回滚所有代码；旧 .bak 也不是可信基线，恢复前核实来源。

```text
lmcache/
  integration/vllm/vllm_v1_adapter.py   scheduler/worker与load/store元数据
  v1/token_database.py                separator、segment、cache key
  v1/compute/blend/
    blender.py                        部分批处理动态切换的入口
    blender_v1.py ... blender_v16.py
    blender_cacheslide.py              V13/V14共享融合
    blender_query_aware.py             V15/V16共享算法
    metadata.py / pyramid.py / utils.py
  v1/compute/models/base.py
benchmarks/rag/
  precompute.py / rag.py / utils.py / output_processing.py
  lmcache_blend.yaml / inputs/
  sbatch/
    lmcache_blender_multi_dataset.sbatch / lmcache_blender_llm_judge.sbatch
    vllm_full_recompute_multi_dataset.sbatch / vllm_full_recompute_llm_judge.sbatch
    blender_v13_14/ / blender_v15_16/   各自multi与judge
    build_answer_tables.py / answer_tables/ / logs/ / 聚合CSV
CacheTune/
  cachetune/                          engine、selection、frequency、rope等
  benchmarks/rag/                     precompute、materialize、rag、metrics、judge
    workloads.json / sbatch/          two_modes、llm_judge
  configs/ / tests/ / README.md
SpecCache/
  bootstrap/sitecustomize.py          隔离overlay
  speccache/                          scorer、selector、alignment、blender overlay
  scripts/render_config.py
  benchmarks/rag/{results,logs,sbatch}/
  configs/ / tests/ / README.md
C2KV/
  scripts/ / sbatch/                  当前主要是数据处理与清洗
  datasets/{raw,processed}/           本地大文件已按要求删除
  python/ / configs/ / benchmarks/ / tests/
  to_do.md / DATASET_FORMAT.md / datasets.md / README.md
tests/v1/                            共用引擎回归测试
Handoff/                             本次交接
```

未列出目录不代表可以删除。

## 3. 模型迁移、主流程和数据集

旧模型 DeepSeek-R1-Distill-Qwen-32B 已迁往 Qwen/Qwen3-32B，不是Qwen3-30B。主实验、full和子实验均涉及迁移，不能只更新一个serve命令。

流程：同一数据集先独立预计算system和各ctx → 保留缓存 → online拼接system、separator、documents、instruction、actual question、answer/chat suffix → 复用与生成 → 清洗答案/静态指标 → judge → CSV/网页。

ctx是独立复用单元。SAMSum ctx通常是完整Dialogue+Summary示例，最终question是待摘要对话；DuReader ctx是证据文章。answers只用于评价，不应把当前目标答案泄露到prompt。

同一数据集precompute和online之间不能清缓存；不同数据集/算法间可以重启服务隔离。较小CPU容量可能引起淘汰和低命中，不等于数据集一定无法执行完。

| Prompt类型 | 任务 | 指标 |
| --- | --- | --- |
| QA | 英文短答案/多跳QA | F1 |
| QA_ZH | multifieldqa_zh | F1 |
| FEW_SHOT | SAMSum对话摘要 | ROUGE-L |
| DUREADER | 中文多文档开放问答 | ROUGE-L |

DuReader的ROUGE-L没错，之前错误在于套用英文对话摘要prompt。已引入DUREADER、QA_ZH中文prompt；SAMSum简洁输出约束也讨论并修改过。当前具体文本以utils.py和CacheTune prompts为准。

query prompt的Question:/问题：后面接actual question，不能误改为Answer。任务类型同时影响构造和指标，修改映射要覆盖主实验、full和子实验。

历史集群文件名：

```text
musique_longbench.json / musique_top50_cacheblend.json / musique_org.json
2wikimqa_top50_cacheblend.json / 2wikimqa_longbench.json / 2wikimqa_org.json
multifieldqa_zh_longbench.json
samsum_top50_cacheblend.json / samsum_longbench.json
dureader_longbench.json / dureader_org.json
```

不要按本地longbench_inputs等目录重写上述集群来源。两份DuReader都应DUREADER。

常见设定：ratio=.16，check_layers=[1]，pyramid=false，QPS=.25，context30000，benchmark输出2048；judge输出4096是另一参数。

**配置差异仍需注意：** 当前本地lmcache_blend.yaml是max_local_cpu_size=400、pyramid=True；用户集群手工文件曾是300。批处理生成临时配置覆盖部分选项，不能声称所有地方已统一300/false。KV_STORAGE_SIZE容量估算和LMCache实际CPU分配不是同一参数，以服务resolved config为准。不要擅自改用户手动模板。

## 4. 已修复的两类缓存问题

### 分隔符与BOS

Qwen3的 `" # # "` 编码为[671,671,220]，没有默认BOS。旧encode(...)[1:]误删真实token，变为[671,220]，导致独立预计算与拼接分段不一致。第一次组合命中0、第二次重复组合命中并不能证明独立缓存复用正确。

token_database.py已使用add_special_tokens=False，不再盲删首token。回归测试见test_token_database.py。

### 长文档chunked prefill存储

precompute.py传 `lmcache.store_full_prompt_on_last_prefill=True`。adapter对这类独立预计算延迟至最后prefill，保存完整prompt；普通chunked行为仍保留。decode判断改为len(token_ids)>prompt_len，避免最后1个prefill token误判。

历史实测：scheduler预算2048，文档6000，Stored6000/6000；组合6366，hit6323，Retrieved6320=system320+doc6000。差3是分隔符位置，不等于doc缓存错位。

另有SAMSum18200/18357 token，lookup18132/18301，retrieved17832/18007。lookup与实际搬运计数不同，须结合mask/separator解释。

LMCache chunk_size256是存储切分，vLLM max_num_batched_tokens是调度预算，两者不同。这个存储修复**不解决V15/V16在线observer Q尚未计算的问题**。

## 5. 输出、TTFT、judge与结果

Qwen3实验关thinking：chat传enable_thinking=False；token completion需在构造chat template时设置。reasoning-parser本身不是关闭开关。

复用实验出现过仅有结束标签的思考泄露；不能断言只发生于nested_union，也不能仅凭输出认定内部根因。

output_processing.py规则：保留raw；取最后一个结束think标签之后作为final；未闭合思考返回空final并标记incomplete；无标签正常保留。保存generated_text、evaluated_text、reasoning_detected、incomplete_reasoning。无法可靠识别所有无标签思考，不用关键词随意删除。

主实验与CacheTune静态指标/judge已接入；SpecCache复用主流程，不需要所有模块都有副本。历史合成plain/full/closing-only/unclosed/multiple测试通过，CacheTune Paris F1=1和judge清洗检查通过，主实验两条真实SAMSum检查通过。

TTFT仍从发请求到首个生成token；不能扣除思考时间。在线小模型打分、加载等也属于在线时间。不要因答案清洗而重定义TTFT。

2048触顶不全是reasoning，还可能是重复或任务prompt问题，prompt限长是软约束。旧“12条触顶、9条judge缺分”只是历史快照，模式/数据集分布需重新统计，本文不编造。

judge上限已改4096；judge_accuracy通常统计score>=.70的比例，average_judge_score是连续分数均值。缺失分数分母需读聚合代码确认。

曾有OOM空答案被judge判1的问题，已增加空/失败预测零分处理及旧记录相关防护。重聚合须核对修复后数据，不能把缺分当正确，也不能拿旧HTML证明修好了。

单次CSV是指标，debug.csv是答案审计，multi_details是逐条汇总，multi_results是任务汇总；judge有独立details/summary。重复跑会产生重复结果，应按run来源核查，不能仅见重复行就删。

生成网页：`python benchmarks/rag/sbatch/build_answer_tables.py`。默认主sbatch结果写answer_tables，支持--sbatch-dir、--output-dir和显式CSV参数，不自动代表所有子实验。本轮未生成/删除网页。

## 6. CacheTune：未解决且用户暂缓的事项

入口README、cachetune/selection.py、engine.py、独立rag。check_layer_fixed与nested_union具体层号/预算以代码为准，不能由名字猜。用户nested_union比例保持不动。

**仍未统一QPS/吞吐。** CacheTune当前同步逐条，QPS不等于主实验异步到达率；throughput为输出tokens/s，主实验为requests/s。用户明确先看质量再做并发改造。这不是已经完成的功能，也未证明是为了OOM才必须串行。

历史job4276633：FAILED、ExitCode9:0、MaxRSS463.99G。单凭这条不能确认内核OOM；9:0也不是signal9。要结合stderr/cgroup/Slurm记录。CPU cache上限与总RSS不同，不能把400改300就宣布根因解决。

## 7. V13/V14加权融合

只实现CacheSlide启发的融合，不是完整动态每4层换token。共享blender_cacheslide.py：alpha为fresh/reuse K差范数除reuse K范数；V13融合K，V14用同alpha融合K/V。默认不clip，alpha可能>1；eps和逐层更新读源码。

用户质疑后层临时加入未选token缺少前层hidden state，因此没有实现动态替换集合，不能把对论文的质疑当成已证明论文错误。

blender_v13_14目录独立multi/judge只跑13/14。部分批处理替换blender.py并通过锁/备份恢复；共享树不能无保护并行替换，强杀后清理前先确认无活跃任务。

## 8. V15/V16 query-aware算法与调度

共享blender_query_aware.py；V15 observer=instruction+question，V16=question，排除answer suffix。唯一消融变量是instruction是否参与observer。

- 完整causal softmax包含system/sep/query等可见key，再读document columns，不能先截document再softmax。
- GQA重复KV heads，对observer和head均值；无pooling、last64、recent强制保留。
- candidate仅复用document；全局document数×ratio预算，非空可至少1。系统不参与repair，但参与attention。
- active=selected docs+mandatory online/cache-miss/separator，不把mandatory算进document16%。
- explicit semantic spans来自prompt构造与request metadata，不猜末尾Ntoken。请求键包括lmcache.blend_prompt_spans。
- full prompt guard保留，不能直接删除。

此前用户要求诊断scheduler→tracker→ReqMeta→start_load_kv→blend→process_qkv，中途额度中断，后来要求还原两个文件的临时诊断改动。没有完整真实V15短/长调度trace，不得声称全面实测。

tracker保存累计已调度前缀，num_computed可能含外部cache hit；can_load控制blend，不是自然延后最后prefill。若首次blend只有prefix，query Q不存在并由guard报错；但prompt长于budget也不必总失败，因为cache hit影响调度覆盖。辅助模型tokens与主runner本轮chunk不能混同。

用户最终要求V15/V16专用multi显式关闭chunking。当前blender_v15_16两条serve路径有max_model_len30000、max_num_batched_tokens默认跟随max_model_len、--no-enable-chunked-prefill。目录另有judge。语法检查通过，不代表GPU显存足够或完整实验已通过。

## 9. SpecCache三种版本

目标32B、小模型0.6B，最后三分之一层hidden state L2范数平均作为importance。token_alignment有兼容检查；同为Qwen或单个hash一样不是所有模板/分词等价证明。

| 入口 | 策略 |
| --- | --- |
| speccache_multi_dataset.sbatch | 原span-aware：整span优先装预算，再单token补位 |
| speccache_multi_dataset_no_span.sbatch | 无span，直接TopK |
| speccache_multi_dataset_paper.sbatch | initial/final分离，扩展后原始importance逐token裁剪 |

三者有对应judge。paper常见initial=.10/final=.16，smooth=.4、coverage=.7、span最长16。当前paper不足预算时还全局补位，这是固定最终预算的实验选择，不应声称论文逐字要求补满。mandatory miss保留，可能使actual_ratio超目标。

原span-aware按span平均importance排序、装得下就整段、再补token，不等于论文单token裁剪。用户要求保留对照，不要混淆。

initial=.16/final=.16且同一ranking裁剪通常退化初始TopK；initial=.10的扩展集合不一定含全局前16%，不必等价直接16%TopK。

overlay的scorer/retrieve/compute_selected_layers都使用传入tokens，actual_ratio=selected/len(tokens)。需沿adapter判断传的是cached prefix还是全prompt，不能套用V15 document-only分母。用户曾接受不额外加入query：causal未来query不改变前面doc hidden state；但候选范围/归一化/预算改变仍可能影响选择。

bootstrap overlay隔离主代码；在线小模型成本计入TTFT。A100两条历史冒烟不是所有变体/长文通过证明。

## 10. C2KV进度、数据和最近修复

先读C2KV/to_do.md、DATASET_FORMAT.md、datasets.md。桌面作者repo与本项目子目录不是同一份。

**完成：目录、数据检查、作者格式转换、LLM清洗脚本与sbatch。未完成：Qwen3 extractor训练/权重、压缩KV持久化与导入、vLLM在线适配、完整benchmark。**

认可的设计：HF冻结base训练gist/C2 embedding和独立投影 → HF按document离线导出压缩KV → CPU/磁盘及元数据 → 原始Qwen3-32B vLLM在线注入、query计算/生成 → 主实验统一测评。

仍需解决GQA、各层布局、dtype、RoPE前后K、压缩位置、mask、paged cache注入与logits等价性。冻结不等于原始权重从未改变，要核对作者projection路由。不能把任意HF cache当普通LMCache segment直接用。离线不计TTFT；在线查找、搬运、位置处理、注入计入。C2KV压缩不自动叠加16%repair。

### 数据

LongMagpie源Seerkfang/LongMagpie_multidoc_longcontext_dataset；HotpotQA；2Wiki源xanhho/2WikiMultihopQA；MuSiQue源dgslibisey/MuSiQue。前三必要，MuSiQue后续用户也要求纳入。

author_format是处理后输入；author_format_dryrun是小样本验证。longmagpie_1024是作者目录命名，不保证每条严格1024个Qwen token，处理包含字符限制，读准备脚本确认。

历史文档统计：LongMagpie428923→428920（空数据3），Hotpot90447/7405，2Wiki167454/12576/12576，MuSiQue19938/2417。本地大数据已删除，不能声称本轮重新统计过。

用户确认全部上传超算后要求删除raw、processed实际数据以及dryrun/.hf_cache，仅保留说明。**这是预期状态，不要重新下载占满新电脑。**

### 清洗

让未压缩模型看文档回答，按参考答案F1过滤，降低base不擅长样本比例；不是证明过滤题永远不可学。OpenAI兼容SDK只是协议，sbatch默认本地Qwen vLLM，不是必须调用付费OpenAI。

当前默认模型32B、port18006、context30000、batched4096、maxseq4、客户端并发8、输出64、F1阈值.5、GPU利用率.70。不要与judge4096混淆。

当前只清hotpotqa/wikimqa/musique，输出author_format_cleaned下对应train_cleaned目录。**用户后来要求四个都清洗，但LongMagpie当前仍排除，需求未完全落实。** 需对照作者格式与实现补齐，不能仅加名字。训练40k配比和MuSiQue纳入仍需按作者入口落实。完成报告跳过与不完整目录防护/断点续跑是不同机制，不能无条件承诺重提交会续跑。

### job4329371失败与修复

日志c2kv_clean_4329371.log只有Missing C2KV root: /var/spool/slurm/C2KV；err空。定位错误发生于服务启动前，不是OOM。

当前sbatch已改：`REPO_ROOT="${REPO_ROOT:-/home/chzhan.chzhipen/LMCache}"`，不从spool定位仓库。bash -n通过；修复后超算重跑尚未确认。

## 11. 本轮发现但没有修改的遗留问题

| 优先级 | 问题/后续 |
| --- | --- |
| 高 | V13/V14和V15/V16独立judge仍由BASH_SOURCE推SCRIPT_DIR/REPO_ROOT；Slurm下有同类路径风险。CSV默认目录也需一起核对，仅覆盖REPO_ROOT可能不够。 |
| 高 | C2KV修复脚本待上传/重跑；四来源清洗尚未覆盖LongMagpie。 |
| 中 | V15/V16聚合quality_metric只把QA标f1，其余rouge_l，QA_ZH标签错误；实际rag分数需独立核对。 |
| 中 | SpecCache paper不足budget补位与论文表述差异需明确。 |
| 中 | 手工YAML400/True与集群300/批处理False不能混为一谈。 |
| 暂缓 | CacheTune并发/QPS/吞吐统一，用户先看质量。 |
| 未做 | C2KV训练与vLLM集成按to_do推进。 |

## 12. 接手验证与命令

相关测试：test_token_database.py的test_segment_token_database_preserves_separator_without_bos；test_vllm_layerwise_wait_for_save.py中的test_full_prompt_precompute_defers_store_until_last_prefill、test_regular_chunked_prefill_still_saves_intermediate_chunks、test_one_token_final_prefill_is_not_treated_as_decode；以及test_blender_query_aware.py。

历史pytest缺cpuinfo导致conftest导入失败，不是算法断言失败。按AGENTS准备依赖，不要直接宣称测试失败等于功能错误。

语法检查不会提交任务：

```bash
bash -n C2KV/sbatch/c2kv_clean_author_datasets.sbatch
bash -n benchmarks/rag/sbatch/blender_v15_16/lmcache_blender_multi_dataset.sbatch
bash -n benchmarks/rag/sbatch/blender_v15_16/lmcache_blender_llm_judge.sbatch
```

用户后续确认环境数据后，可在超算提交清洗：

```bash
cd /home/chzhan.chzhipen/LMCache
mkdir -p C2KV/sbatch/logs
sbatch C2KV/sbatch/c2kv_clean_author_datasets.sbatch
```

本轮没有远端提交，没有改算法，没有删除数据。新Codex根据用户下一项任务推进，修改前核查源码，完成后交代具体文件、验证结果和仍未实测的边界。
