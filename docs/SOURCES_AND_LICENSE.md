# 来源、代码复用与许可初查

## 方法参考与当前代码

主要参考[langchain-ai/rag-from-scratch](https://github.com/langchain-ai/rag-from-scratch)的[1_to_4教学Notebook](https://github.com/langchain-ai/rag-from-scratch/blob/1fdb7d0e4f1d3e731c8397332452715512f18667/rag_from_scratch_1_to_4.ipynb)，学习索引、检索、生成流程，不是完整复现该仓库。当前六模块为长期ChatGPT辅助迭代交付，未附原Notebook、图片或其输出。

当前六个模块与该固定Notebook代码单元做了去空白行/注释行的逐行字面比较，未发现两个连续有效行一致的段落；最长单个文件一致段为0～1行，见../evidence/literal_code_comparison.json。这不是版权判断，不检测改写/改名，也不能证明历史上完全没有复制。作者于2026-11-06补充说明：六个拟发布模块均直接替换助手提供的下载文件，未另报告手动复制其他仓库具体代码。这是作者来源陈述，不是对AI辅助代码的完全原创或第三方权利保证。上游方法参考仍明确保留。

## 上游许可证初查

2026-11-06通过GitHub连接读取main，提交为1fdb7d0e4f1d3e731c8397332452715512f18667。根目录包含README和5个Notebook，未发现LICENSE/COPYING；README没有可确认的许可声明。许可证缺失[Issue27](https://github.com/langchain-ai/rag-from-scratch/issues/27)仍Open，未见维护者给出的许可证答复。可复核资料见../evidence/upstream_check.json。

本项目不默认MIT，也不套用LangChain其他仓库许可证。本包没有替作者选择个人改造代码许可证，没有附LICENSE来冒充已获上游授权。GitHub官方说明：[Licensing a repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository)。许可未确认与未选择个人代码许可分别记录，不能把“公开可见”当作所有复用均已获授权。

本次检查是来源初查，不是所有历史复用和124项依赖的完整许可审计。公开前按实际复用内容逐项核对，有直接复用时保留适用声明与授权，不能用一份许可覆盖第三方代码和资料。未向上游发消息或代用户申请授权。

## 官方说明书

固定中文V1.0，不宣称是最新资料。每份15页，参数表PDF第4页。只提供来源与指纹，不分发PDF、产品图片或整段候选正文。

| 产品/本地文件 | 字节数 | SHA256 |
| --- | ---: | --- |
| RIVER 2 / river2_manual_cn.pdf.pdf | 863948 | 2cacfb6b780d1242398bc0879ec3a328741505c4ca7d7b948bd867e9038de428 |
| Max / river2_max_manual_cn.pdf | 904783 | 5d37a1710770517f135abe053c23b266eac7af07f1076d3710f20cbdb2254157 |

- [RIVER 2官方中文PDF](https://websiteoss.ecoflow.com/cms/upload/2023/1/13/ECOFLOW%20RIVER%202%20-%20%E7%94%A8%E6%88%B7%E6%89%8B%E5%86%8C%20%28%E4%B8%AD%E6%96%87%29%20V1.0_1673603646202.pdf)
- [Max官方中文PDF](https://websiteoss.ecoflow.com/cms/upload/2023/1/13/EcoFlow%20RIVER%202%20Max-%E7%94%A8%E6%88%B7%E6%89%8B%E5%86%8C%20%28CHS%29%20V1.0_1673603717374.pdf)

链接记录既有资料来源，本次没有重新下载PDF验证远端字节；下载后可按哈希核对版本。不因上传知识库就取得资料版权。

examples/demo_product_test.txt为虚构TEST-A，用于功能测试；不属于真实产品、售价或商业使用证据。

## 第三方依赖

requirements.windows.actual.txt来自用户已运行Windows环境，124项固定版本；requirements.txt仅去BOM/规范换行。代码直接使用streamlit、python-dotenv、langchain-community、langchain-text-splitters、langchain-core、langchain-openai、langchain-chroma、chromadb、jieba、rank-bm25、pypdf。依赖仍受各自许可证约束，没有将它们的源代码复制进本包。完整依赖许可审计未完成，不声称已完成。
