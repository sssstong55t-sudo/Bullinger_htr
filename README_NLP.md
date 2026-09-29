# Bullinger NLP：可检查的第一版基线

这是一套独立代码，不修改 bullinger-htr，不训练或加载 HTR 模型，不上传数据，不调用付费 API。Python 3.10 或更新版本即可，无第三方依赖。

## 这版是什么

用可编辑地名词表查找文本中的名称，再查询候选地点。保留原文、字符位置、来源行和已有坐标，供后端、地图和原稿查看器对接。

这是 **gazetteer baseline（词表基线）**，不是训练好的历史拉丁语/德语 NER，也不具备上下文消歧能力。第一版的意义是建立接口和可重复评估，不是宣称已经完成 NLP。示例词表只有三个城市，不能代表 Bullinger 地名覆盖率。没有命中并不代表文本没有地名。

不会把同名地点强行合并；多个候选返回 ambiguous。只有一个候选也只是 unique_in_gazetteer（在当前词表中唯一），不等于历史语境中已确认。所有结果 needs_review=true，confidence=null。

## 文件

- nlp_baseline.py：提取、候选匹配、评估、可导入的 Python 类。
- export_transcriptions.py：读取你已有 medium_split.json，导出 TXT 人工参考文本。
- gazetteer.example.json：极小演示词表，城市点坐标仅供演示。
- demo_reference.jsonl / demo_htr.jsonl：人工构造的示例，不是你模型的真实输出。
- test_nlp.py：偏移、边界、歧义、评估计数等测试。

## 先运行演示

在解压后的本目录运行；Windows 可以把 python3 改为 python。

```bash
python3 nlp_baseline.py extract --input demo_reference.jsonl --gazetteer gazetteer.example.json --output demo_reference_predictions.jsonl
python3 nlp_baseline.py extract --input demo_htr.jsonl --gazetteer gazetteer.example.json --output demo_htr_predictions.jsonl
python3 nlp_baseline.py evaluate --gold demo_reference.jsonl --predictions demo_reference_predictions.jsonl --mode span --output demo_span_scores.json
python3 nlp_baseline.py evaluate --gold demo_reference.jsonl --predictions demo_reference_predictions.jsonl --mode places --output demo_reference_place_scores.json
python3 nlp_baseline.py evaluate --gold demo_reference.jsonl --predictions demo_htr_predictions.jsonl --mode places --output demo_htr_place_scores.json
python3 -m unittest -v
```

## 接入你现在的数据

先在这个独立文件夹运行，输出也留在这里：

```bash
python3 export_transcriptions.py --split-file /Users/shaotongsmac/Desktop/bullinger-htr/medium_split.json --split validation --output bullinger_reference.jsonl
python3 nlp_baseline.py extract --input bullinger_reference.jsonl --gazetteer gazetteer.example.json --output bullinger_reference_predictions.jsonl
```

这里使用 validation 做开发，未修复现有按行划分；不是新的独立测试集。需要正式结论时另建按信件隔离的数据划分。原始 TXT 来自自动对齐数据，不保证完全正确，人工标注前应核对行图。导出程序不会编造 gold_mentions，因此不能直接把导出文件用于有监督评估。

## 输入：每行一个 JSON 对象

```json
{"document_id":"6","page_id":"6_00","line_id":"6_00_r1l5","language":"la","text":"Turicum","image_path":"/path/to/line.png","line_bbox":null,"text_source":"dataset_reference"}
```

document_id、page_id、line_id、text 必须为字符串。每组 ID 唯一。language 可为 la/de。保持所有文本原样传递，尤其不要在预测完成后改变空格，否则 offsets 会失效。

有真实坐标才提供 line_bbox，例如 {"x":10,"y":20,"width":300,"height":40,"page_width":1200,"page_height":1800}。这些值应对应原页像素坐标，不是裁剪图坐标。模块只透传，不推断坐标；框的精度仅到行，不能称为地名词框。不要将包含本地 image_path 的内部结果直接公开给浏览器，后端应转换为受控图片 ID/URL。

## 接入 medium 预测

在你自己选择采用的推理代码中，对同一条记录替换 text 为模型输出，并设置 text_source="htr"。保留 document_id/page_id/line_id；另存 htr_input.jsonl，再执行 extract。

不要直接 import train_medium.py：原脚本含有顶层训练逻辑。这个代码包不会导入它，也不会自动运行训练。medium 推理本身仍由你现有 HTR 流程负责。

从其他 Python 模块调用：

```python
from nlp_baseline import GazetteerNLP, read_json
nlp = GazetteerNLP(read_json("gazetteer.example.json"))
result = nlp.extract({"document_id":"6", "page_id":"6_00", "line_id":"6_00_r1l5", "text":"Turicum"})
```

在服务启动时初始化一次 nlp，之后复用。这个模块不是 HTTP 服务；可以由你们的 FastAPI service 调用。

## 标注和评估

给人工核查后的参考行增加 gold_mentions：

```json
"gold_mentions": [{"start": 0, "end": 7, "place_id": "zurich"}]
```

start 从 0 开始，end 不包含在范围中；按 Python 字符索引，不是 UTF-8 字节。上述范围对应 "Turicum"。无地名的已核查行必须明确写 []，未标注行不要填 []。地名存在但实际地点不确定时 place_id=null。标注整个评估样本，包括词表未覆盖的地点；不要直接拿程序提取结果当金标准。

两种评估：

1. **span**：对同一份文本检查精确起止位置，输出 micro precision/recall/F1，并按语言分组。输入文本不一致会报错，不允许用人工转写的字符位置直接评价 OCR 文本。若要评价 HTR 上的纯 NER span F1，须另行标注 HTR 文本本身的实体位置。
2. **places**：按来源行比较标准 place_id 的出现次数，重复出现会重复计数。不比较字符位置，适合人工转写与 HTR 对照。这是端到端地点提及恢复指标，**不是 NER span F1，也不是单独的地理消歧准确率**。含未知金标准地点的整行被排除，报告排除数；应同时报告覆盖范围。候选歧义没有确定 place_id 时不计作成功恢复。跨行移动或合并需要先对齐来源行，本版本不自动对齐。

报告示例中的分数仅是合成测试结果，不能作为实际模型质量。没有任何实体且分母为零时指标按 0 返回，结合 tp/fp/fn 理解。不同语言的指标按金标准行的 language 分组。

## 如何扩充词表

复制 gazetteer.example.json 为你的版本。每个地点增加稳定 id、标准名、真实来源、经核验坐标和明确别名；同一地点多个拼写放同一条 aliases。不同地点同名时，在不同地点各自添加相同别名，程序会保留歧义。拉丁语变格形式必须核查后显式添加，程序不会自动推导。新增地点不需要改 Python 代码。

历史别名 Turicum 的示例依据：苏黎世州官方资料 https://www.zh.ch/de/news-uebersicht/medienmitteilungen/2021/02/turicum-eine-roemische-hafenstadt-und-zollstation.html 。这只支持历史名称对应，不表示已经覆盖 Bullinger 时期全部词形。示例坐标是近似现代城市点，不是历史边界或历史精确位置。

在训练/开发样本上维护词表，固定词表版本后再用独立测试集评估。不要看到测试漏检就补词再报告同一测试分数。

## 已知边界与下一步

- 缺少上下文：人名、书名中的同形名称可能误报；大小写不敏感也会导致误报。
- 不自动纠正 HTR 错字，不做模糊匹配，避免把近似词偷偷当成地点。
- 不处理跨行地名、断行连字符、多词名称间空白差异；只支持词表精确形式和字符级 NFKC/casefold。分解 Unicode 重音形式不保证与预组合字符等价。
- 没有外部地理编码、NER 训练、地图、整页分行或 calibrated confidence。
- 下一阶段：用人工标注比较词表与适合语料的 NER 模型；再做上下文消歧，并分别报告 NER、地理匹配和端到端结果。

先建立可复现基线，再由实际错误决定是否需要训练 NER、扩充历史词形，或继续优化 HTR。
