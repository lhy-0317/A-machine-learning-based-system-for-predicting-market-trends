# 基于机器学习的市场趋势预测系统

第13队：刘瀚远、韩启源、曾玺玮、杜凌琳。课程提交整理日期：2026-10-23。

## 项目范围

本项目是 PyTorch MLP + Flask + Jinja2/Chart.js 的市场趋势分类课程原型，支持单条或批量JSON输入、三类标签与概率输出、训练曲线和健康查询。

数据由他人提供，原始来源与生成流程待核实；仅作为课程演示数据。原13维输入包含 future_7d_return，存在前视信息泄漏。保存模型的97.25%验证准确率不等于真实未来市场预测能力。删除该字段后的补充随机分层实验为81.79%，盘整召回率为0。时间划分末291条全为上涨，98.63%准确率低于恒预测上涨的100%。详见技术报告和verification结果。

## 安装与启动

在项目根目录执行（Windows PowerShell示例）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m flask --app app run --host 127.0.0.1 --port 5001
```

浏览器打开 http://127.0.0.1:5001 。已附带模型，无需先训练。Chart.js通过CDN加载，图表需要网络可达。

`requirements.txt`为原始未锁定依赖；`requirements-reproduced.txt`记录本次实测主要版本。若需要匹配复核环境，使用Python 3.8.20及该锁定文件；CPU版PyTorch可单独安装：

```powershell
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-reproduced.txt
```

本次复核是在已有环境完成，不声称已验证全新环境安装。原Scaler持久化版本为scikit-learn 1.3.0，本次1.3.2会产生版本提示。`venv`、`.idea`及`__pycache__`没有打包；原虚拟环境绑定他人电脑路径，应自行重建。

## API使用

```powershell
$metadata = Get-Content .\models\model_metadata.json -Raw | ConvertFrom-Json
$body = $metadata.default_payload | ConvertTo-Json
Invoke-RestMethod -Uri http://127.0.0.1:5001/predict -Method Post -ContentType 'application/json' -Body $body
```

- GET `/`：表单页面。
- GET `/health`：特征列表和类别。
- GET `/status`：服务描述。
- GET `/metrics`：50轮训练历史。
- POST `/predict`：对象输入返回对象；对象数组返回数组。

字段顺序由model_metadata.json确定。默认样例包含未来收益率，仅用于复现旧原型，不是实时预测输入。

## 复核与补充实验

```powershell
python verification/verify_project.py
```

脚本使用CPU和2个PyTorch线程，重建random_state=2024的原始8:2分层划分，评估保存模型并执行17个API场景。随后以随机种子2024进行13维随机划分、12维随机划分、12维时间划分加7条间隔三组诊断训练，每组50轮，深复制最佳状态，最后更新verification_results.json。它不修改models目录。补充实验验证集用于选轮，未提供独立测试集，也未进行多种子统计。

原始代码与模型未作功能性修改；新增README、版本记录、verification脚本和结果均为本次提交整理与复核材料。

## 已知限制

- 原训练输入存在未来收益率泄漏；其余指标时间可获得性也待核实。
- `best_state = model.state_dict()`没有深复制，不能可靠保存最佳状态；无真实早停。
- 原损失没有类别权重；没有SHAP、收益回归、交易回测或NLP模块。
- 空数组、Infinity、float32溢出会产生500；负成交量未拒绝；NaN可进入结果。正常演示请使用有限且符合语义的数值。
- 原app.py直接启动时监听0.0.0.0并启用debug；演示建议使用上述flask命令绑定127.0.0.1。
- 概率显示为标签，未做校准；训练图表依赖CDN，无内置缩放插件。
- 不支持跨平台位级重现保证，延迟结果只包含进程内测试客户端调用，不含浏览器或网络。

## 原模型重训

`python train_model.py`会覆盖models下的权重、标准化器、元数据和训练历史，重训前请先备份。原脚本未固定全部随机性，结果可能变化，且不会自动修复上述研究设计问题。

## 目录

app.py / train_model.py：原应用与训练代码
templates/：原网页模板
data/：1453条演示数据
models/：原权重、Scaler、元数据和50轮日志
reports/：原界面及历史曲线图片，历史曲线可能与当前JSON不一致
verification/：新增复核脚本、结果、环境记录

技术报告以当前JSON及复核结果为定量依据，历史文档的近似指标不作为新结论。核心原始资产的SHA256保存在verification_results.json中。
