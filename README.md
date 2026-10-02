# 番茄病虫害检测演示系统

基于已训练完成的 P2-DySample 和 GT-RFSKD 模型，提供图片上传、检测框可视化、分类计数、批量检测和 CSV 导出。网页端采用 Streamlit，推理端采用 ONNX Runtime CPU，不依赖训练服务器或显卡。

本项目对应研究课题《基于改进 YOLOv11 的复杂背景番茄病虫害检测与轻量化方法研究及系统实现》。网页演示不用于植保处方，也不能替代专业诊断。

## 本地运行

使用 Python 3.12。在本目录中运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python verify_models.py
python -m unittest discover -s tests -v
streamlit run app.py
```

随后访问 `http://localhost:8501`。默认加载 GT-RFSKD。高精度模型可在内存测试通过后，通过环境变量开启：

```bash
ENABLE_P2D=1 streamlit run app.py
```

网页采用常见的系统无衬线字体。中文优先使用设备上的苹方、微软雅黑或 Noto Sans CJK，英文字母使用系统界面字体。没有将商业字体文件随仓库分发。该选择只适用于网页，不改变论文和答辩文件的字体规范。

## Render 部署

1. 将本目录作为独立 GitHub 仓库上传，不要上传论文工作区。
2. 登录 [Render](https://dashboard.render.com/)，授权访问该仓库。
3. 新建 Blueprint，选中仓库，读取根目录 `render.yaml`。
4. 核对服务类型为 Web Service、实例为 **Free**，不要添加数据库、付费磁盘或其他服务。
5. 填写 `DEMO_ACCESS_CODE` 作为演示访问口令。只在 Render 的环境变量中填写，不写进仓库。`SOURCE_URL` 填本仓库的 GitHub 地址。
6. 部署完成后检查 `/_stcore/health`，再进入页面上传图片。至少完成一次真实检测和结果导出后才视为上线成功。

也可手动创建 Web Service。构建命令为 `pip install -r requirements.txt && python verify_models.py`，启动命令为 `streamlit run app.py --server.address 0.0.0.0 --server.port $PORT`，健康检查路径为 `/_stcore/health`。

Render 提供的是 `服务名.onrender.com` 子域名，不是免费赠送独立域名。免费实例有休眠和资源限制，首次访问可能需要等待启动。当前配置关闭自动部署，更新代码后手动触发部署，避免不必要的构建消耗。

免费服务不等于绝无收费可能。若账户绑定付款方式，超过包含的带宽或构建用量可能产生费用。部署前应查看账户的用量与支出限制，不要启用自动升级。平台规则以 [免费服务说明](https://render.com/docs/free) 和 [服务配置文档](https://render.com/docs/blueprint-spec) 为准。

## 输入与输出

- JPEG、PNG；每批最多两张，每张最大 2 MiB、400 万像素。
- 输入经 EXIF 方向修正、RGB 转换、等比例缩放和边缘填充后，形成 `1×3×640×640` 张量。
- 采用分类别 NMS，默认置信度 0.25、NMS IoU 0.70，最多保留 300 个目标。
- 页面显示原图与检测图，明细给出原图坐标、类别和置信度。无检出时仍能导出含表头的 CSV。
- ZIP 内包含检测预览图及 CSV。预览图长边不超过 1280 像素；CSV 坐标对应经 EXIF 修正后的原图，而非缩略图。
- 页面推理耗时是 ONNX Runtime 前向时间。预处理和后处理单独列出，不包含网络传输、图片绘制、下载或浏览器渲染，不应与论文中的 RTX 4070 延迟混写。

## 模型

|模型|参数量|ONNX 大小|输入|检测输出|
|---|---:|---:|---|---|
|GT-RFSKD|2,591,010|10.08 MiB|1×3×640×640|1×10×8400|
|P2-DySample|9,604,840|37.32 MiB|1×3×640×640|1×10×34000|

两个 ONNX 是同一研究中的最终 seed 0 导出文件，没有为网页修改权重或量化。模型 SHA-256、字节数、输入输出形状保存在 `models/manifest.json`，构建时校验。GT-RFSKD 推理图不包含教师或训练期适配器。

类别顺序为晚疫病、潜叶虫害、缺镁、缺氮、缺钾和斑萎病毒病。英文 `Pottassium_Deficiency` 沿用训练数据原有拼写，不在部署时修改类别索引。

模型仅在研究使用的固定训练与验证划分上评估，尚无独立外部测试。历史 GT-RFSKD 长程训练使用过验证集类别计数，而且训练预算高于无蒸馏基线。因此不能仅凭该模型的既有分数宣称等预算、无验证信息参与的独立蒸馏收益。网页不重新发布排行榜或泛化承诺。

## 与本地研究系统的区别

本地研究系统使用 PyTorch CUDA，可显示中间层特征活化。这里使用仅输出检测响应的 ONNX，不提供 Grad-CAM 或特征活化图，避免将检测置信度图冒充解释性可视化。论文中的 GPU 测速、热力图和历史界面测试仍属于本地研究系统，不能当作本网页的已完成实验。

免费实例默认只开放轻量模型，高精度模型不是默认演示选项。`ENABLE_P2D=1` 可开启切换，但应先确认实例内存足够。两类模型串行运行，同一进程最多保留一个推理会话。公共服务仍不保证不限人数、不限并发。

## 数据与安全

本仓库不包含训练数据、论文、服务器地址、SSH 配置、账号口令或训练日志。上传图片仅在当前进程中处理，不写入持久磁盘，不用于训练；用户图片不会提交到 GitHub。浏览器、Streamlit 及托管平台仍可能短期缓存会话内容，因此不要上传敏感照片。公网展示建议保留访问口令，并只提供给答辩或演示参与者。

当前只设置进程级串行推理、文件大小与像素限制、格式验证和 CSV 公式转义。它是低并发研究演示，不是经过渗透测试的生产服务。不要将它用于高并发公开业务。

## 文件说明

|文件|作用|
|---|---|
|`app.py`|上传、显示、导出与访问口令|
|`inference.py`|输入处理、ONNX 推理、NMS、坐标还原和绘图|
|`models/`|最终 ONNX 和校验清单|
|`verify_models.py`|构建时检查权重完整性|
|`tests/test_inference.py`|输入边界、坐标恢复、NMS 和 CSV 测试|
|`requirements.txt`|部署依赖|
|`.streamlit/config.toml`|界面、上传与跨站保护设置|
|`render.yaml`|免费 Web Service 配置|

## 许可与来源

本项目代码按 AGPL-3.0 发布。模型导出自 Ultralytics YOLO11 实现，导出文件内声明 AGPL-3.0；使用或再发布时应保留相应声明。将模型转换成 ONNX 不改变原有许可义务。参见 [Ultralytics 许可说明](https://www.ultralytics.com/license)。

- [Ultralytics](https://github.com/ultralytics/ultralytics)，YOLO11 基础实现。
- [DySample](https://github.com/tiny-smart/dysample)，动态上采样方法与公开实现。
- [Tomato-Village](https://github.com/mamta-joshi-gehlot/Tomato-Village)，研究使用的数据来源，本仓库不重新分发数据。

第三方依赖保留各自许可。商业用途、闭源分发及数据使用范围需由使用方进一步确认，不应将本项目说明视为法律意见。
