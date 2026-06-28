<img width="1604" height="333" alt="image" src="https://github.com/user-attachments/assets/b4813f37-6a84-4764-8866-9ee07dcb5456" /># 人脸识别自动考勤系统

基于 InsightFace + FastAPI 构建的离线人脸识别考勤系统，实现人脸注册、1:1 身份核验、1:N 全库检索、无感自动打卡与考勤记录全链路管理。采用前后端算力分离设计，兼顾识别精度与部署便捷性。

## 技术栈
| 层级     | 技术选型                                                 |
| -------- | -------------------------------------------------------- |
| 前端     | 原生 JavaScript + face-api.js + WebRTC + Canvas          |
| 后端框架 | FastAPI + Uvicorn ASGI 服务器                            |
| 算法引擎 | InsightFace (ArcFace) + OpenCV + NumPy + ONNX Runtime    |
| 数据层   | SQLAlchemy ORM，支持 SQLite / MySQL 双模式                |
| 工程规范 | 全局统一异常处理 + 标准化返回体 + 依赖注入 + 事务回滚     |
| 部署兼容 | CPU/CUDA 自适应推理，支持 x86/ARM 架构                   |

## 核心功能

### 人脸算法能力  
- 人脸注册：自动检测人脸并提取 512 维特征向量入库，拦截重复注册  
- 1:1 人脸核验：指定用户 ID 进行精准身份比对，适用于登录、权限校验场景  
- 1:N 人脸检索：单张人脸自动与全库用户批量比对，输出最优匹配结果  
- 离线测试脚本：脱离 Web 服务即可独立验证特征提取与比对逻辑  

### 考勤业务能力  
- 无感自动打卡：人脸检测-特征匹配-记录生成全流程闭环，支持上班/下班双类型  
- 防误触机制：连续帧校验 + 打卡冷却时间，避免画面抖动、路人误入产生无效记录  
- 考勤记录查询：支持最近记录分页查询、指定用户历史记录全量追溯  

### 管理运维能力  
- 用户全生命周期管理：注册、查询、级联删除（同步清理对应打卡记录）  
- 管理员鉴权：请求头 Token 校验，非授权用户无法访问管理接口  
- 双数据库兼容：默认 SQLite 开箱即用，可一键切换 MySQL 适配生产环境  

### 前端交互能力  
- 浏览器端人脸预检测：本地完成人脸画框预览，仅稳定人脸触发后端请求  
- 实时状态反馈：摄像头状态、识别结果、操作提示全链路可视化  
- 兼容适配：适配低版本浏览器 Canvas API，支持设备权限管控  

## 关键技术实现

1. **高精度人脸识别**  
采用 InsightFace buffalo_l 模型套件，基于 ArcFace 算法输出 512 维归一化特征向量，使用余弦相似度完成比对；为 1:1 核验与 1:N 检索设置差异化阈值，平衡准确率与召回率。

2. **向量化 1:N 检索优化**  
基于 NumPy 构建特征矩阵，通过单次矩阵运算完成全库相似度计算，替代循环遍历方案，在大用户量下显著提升检索效率。

3. **工程化数据库设计**  
采用「用户主表 + 考勤流水表」分层设计，高频查询字段建立索引；通过 FastAPI 依赖注入管理数据库会话生命周期，请求结束自动释放；所有写入操作配套异常捕获与事务回滚，保障数据一致性。

4. **前后端算力分离架构**  
前端通过 face-api.js 完成本地人脸检测与稳定性判断，仅稳定人脸发起请求，有效降低服务端算力消耗；配合帧校验与冷却时间双重规则，避免重复、误触发打卡。

5. **部署兼容与工程规范**  
ONNX Runtime 自适应 CPU/CUDA 推理，有 GPU 自动启用加速；全局统一异常拦截，返回标准化 JSON 格式；配置与代码分离，敏感信息通过环境变量管理；后端大模型不入库，配套一键下载脚本自动部署。

## 快速开始

### 环境要求
- Python 3.8+
- （可选）NVIDIA GPU + CUDA 环境，用于推理加速

### 部署步骤

1. **克隆仓库**
```bash
git clone https://github.com/Yolande1994/face_recognition_attendance_system.git
cd face_recognition_attendance_system
```

2.**安装依赖**
```bash
python -m venv venv
# Windows 激活虚拟环境
venv\Scripts\activate
pip install -r requirements.txt
```

3.**下载后端人脸模型**
```bash
python download_models.py
```
前端 face-api.js 模型已内置在 public/models/ 中，无需额外下载。

4.**配置环境（可选）**
复制 .env.example 为 .env，可修改端口、识别阈值、数据库类型、管理员 Token 等配置；默认使用 SQLite 数据库，无需额外配置即可启动。

5.**启动服务**
```bash
python main.py
```

## 项目目录结构
face_recognition_attendance_system/
├── algorithm/ # 人脸算法引擎封装
├── core/ # 全局配置、统一异常处理
├── database/ # ORM 模型、数据库 CRUD 封装
├── schemas/ # 接口数据模型定义
├── utils/ # 通用工具函数
├── public/ # 前端静态资源与模型权重
├── .env.example # 环境配置模板
├── .gitignore # Git 忽略规则
├── download_models.py # 后端模型一键部署脚本
├── test_algorithm.py # 离线算法测试脚本
├── main.py # 服务启动入口
├── requirements.txt # 项目依赖清单
└── index.html # 前端演示页面
