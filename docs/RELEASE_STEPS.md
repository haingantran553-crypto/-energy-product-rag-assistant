# GitHub发布与后续维护

当前状态：独立Windows虚拟环境验收通过；当前运行模块冻结；作者已补充六模块直接替换助手交付文件的来源说明，个人代码许可证尚未选择；仓库已创建并上传代码，以下保留初次发布流程供参考，已有仓库不用重复创建。

1. 将本包解压到独立目录，不覆盖原rag-from-scratch下载目录。不把原仓库Notebook、.venv、_local_kb、.env、原始评测、PDF和备份一起上传。
2. 来源说明已经记录作者确认，见SOURCES_AND_LICENSE.md。原目录备份、知识库、历史全部保留本机；不要复制到发布目录。方法参考与依赖来源继续保留，不虚称全部原创，不默认MIT。
3. 检查README中的作者信息、事实、已完成/待完成边界。当前无真实生产用户、没有v6.6.1完整20题率。
4. 核对发布范围和来源说明后，在GitHub新建空仓库；仓库名可用energy-product-rag-assistant。不要初始化另一份README/LICENSE造成冲突。实际许可选择根据已确认范围另行决定。
5. 在本包解压目录的PowerShell执行以下本地Git步骤，查看暂存清单后才推送：

```powershell
git init
git branch -M main
git add .
git diff --cached --name-only
git status
```

核对清单不含.env、_local_kb、虚拟环境、PDF、SQLite、qa_records原始文件或Notebook。配置作者身份时使用自己的Git身份；不要把用户名/邮箱写成虚构值。

```powershell
git commit -m "Prepare energy product RAG prototype v6.6.1"
```

当前仓库：https://github.com/haingantran553-crypto/-energy-product-rag-assistant 。采用Git方式维护时使用该真实地址；网页更新无需执行git remote或push命令。

公开后的验收应从仓库链接检查文件与README；本包自检不代表已经公开成功。若后续增加资料/用户数据，再次核对.gitignore与git暂存清单，忽略规则不能撤回已经提交的文件。

网页手动上传也只选择本干净发布目录中的文件，不依赖.gitignore替你筛选原目录内容。若目录里意外混入私密资料，先移出待上传范围并核对清单，不删除唯一原件。
