# Generative AI usage record

Student: Peilin Yang, 25021827

Tools: OpenAI Codex (generation and first verification; earlier responses identified their model as GPT-5, the regeneration turn as GPT-6 according to the session configuration); WorkBuddy (a separate AI assistant, used for the independent re-verification and final packaging session recorded in section 6, and for the 16,000-row V3 regeneration recorded in section 7). These names describe the sessions, not independently verified model telemetry.

## Scope of assistance

Generative AI assistance was extensive: reading the supplied requirements, drafting and revising C++ source, building the Qt interface, writing automated tests, running experiments, preparing Excel workbooks, and drafting this record and the report. Assistance was not limited to planning and debugging. The student has not stated that they independently wrote, reviewed or verified these generated files; no such declaration is made here.

The checklist says AI may be used for planning and debugging. This record discloses the broader assistance actually used. The student should confirm that this use complies with the course's permitted scope before submitting. This document is not a student signature or an assertion of instructor approval.

## Prompts and responses relevant to the deliverables

The following prompts are copied from the conversation. Response excerpts are labelled as excerpts; summaries are not represented as quotations. Installation screenshots, cookie dialogs and account setup messages are omitted because they did not change the assignment algorithms or data. This is a selected conversation record, not an exhaustive transcript.

### 1. Original generation

User prompt: "请帮我完成这个作业" (Please help me complete this assignment), with the assignment PDF, starter ZIP and two workbook templates attached.

Assistant response excerpt: "我会先读取作业说明、启动代码和两份表格，明确评分要求与已有进度；然后在工作区完成代码、运行验证，并整理可提交的结果。"

Generated response/output summary: an initial 16-64-32-26 tanh C++ model, a separate Python training script, saved weights with 91.925% test PGC, Excel workbooks, a short user guide and an AI declaration. That draft had defects: its GUI read training data in place of test/validation data, its C++ and Python training procedures differed, and checklist entries were written into wrong rows. These outputs were superseded by V2 and are not included as final results.

### 2. AI declaration and student details

User prompt: "是否有AL声明" (Is there an AI declaration?).

Assistant response excerpt: "有，压缩包内已包含 `AI_USAGE_DECLARATION.md`。"

User prompt: "我的名字是Peilin Yang .学号是25021827".

Assistant response/output summary: the earlier draft attempted to add Yang / Peilin / 25021827 to the checklist. The later review discovered that these values had overwritten the header row. V2 restores the source template and puts student details into D5:F5.

### 3. Compilation help

User prompts: "请帮我下载一个可以编译的软件", "请帮我把代码运行", and "怎么安装".

Assistant response/output summary: guidance to install Qt Creator, Qt libraries and a matching MinGW toolchain. The screenshots showed Qt Creator 20.0.1, Qt 6.11.2 and MinGW 13.1.0. Subsequent command-line builds used these installed tools.

### 4. Review of the draft

User prompt: "关于我的作业 重新查看有没有改进的地方" (Review my assignment again for possible improvements).

Assistant response excerpts:

"有，而且有几处必须修正。我之前说‘可以直接提交’太早了，抱歉。"

"这次已实际确认：Qt 已安装，源码编译成功；用保存的权重重新计算，测试准确率确实是 91.925%，混淆矩阵 CSV 也一致。"

Response summary: the review identified wrong dataset paths, misaligned checklist fields, inconsistent training implementations, absent ReLU/configuration experiments, missing test curves, missing target-PGC stopping and test-file selection, lost matrix headers, and incomplete documentation. It recommended correcting these before submission.

### 5. Regeneration (V2)

User prompt: "重新生成" (Regenerate).

Assistant response excerpt: "我会重新生成提交包：修复数据读取和表格错位，补上 ReLU/tanh 对比实验，并用同一套 C++ 网络引擎完成训练、测试和 Qt 界面运行。"

Assistant progress response excerpt: "六组实验已完成：按验证集 MCEL 选出的最佳网络是 `16–64–32–26`、ReLU，保存在第 143 轮。它在 4,000 条测试数据上答对 3,752 条，准确率为 93.80%，MCEL 为 0.203595。"

Generated response/output summary: the files in this V2 archive are the substantive generated source and document outputs for this request. In particular, `backpropagation.cpp` contains the implemented forward/backward calculations, `experiments.cpp` contains the tested experimental procedure, and `mainwindow.cpp` contains the Qt interface. `REPORT.pdf`, the workbooks and the logs record the measured results. The AI invoked real local compilers and the C++ runner; the numerical results are program outputs, not estimates invented in prose.

### 6. Independent re-verification and packaging (WorkBuddy session, 7 September 2026)

User prompt: the review findings listed at the top of this conversation turn (wrong dataset paths, misaligned checklist rows, training-implementation mismatch, missing comparison experiments and test curves, missing PGC stop and test-file selection, lost confusion headers, unnumbered configuration file, and documentation gaps), with the request to fix the submission package accordingly.

Assistant response summary: rather than re-running training, this session independently re-verified the already-generated V2 package and then packaged it. It (1) confirmed by code inspection that `mainwindow.cpp` reads `training_set.csv`, `validation_set.csv` and `test_set.csv` separately, offers per-split file-selection buttons, stops training at a target training PGC, reports load failures as ERROR messages, and writes run outputs only under `gui_runs/` so the submitted `best_weights.txt` is never overwritten; (2) diffed `checklist.xlsx` cell-by-cell against the previous defective checklist and confirmed the header row is restored and student details sit in D5:F5 with every answer in its template row; (3) rebuilt `experiments.exe` and the Qt GUI from the packaged sources using the installed Qt 6.11.2 / MinGW GCC 13.1.0 toolchain (both compiled without errors or warnings); (4) re-ran `experiments --self-test` (all checks passed; finite-difference gradient errors about 4e-10); (5) wrote an independent loader for the packaged `best_weights.txt` and recomputed all three splits - training 12,406/12,800, validation 3,012/3,200, test 3,752/4,000 (PGC 93.800%, MCEL 0.20359454) - matching `best_result.txt` exactly, with zero confusion-matrix mismatches against `confusion_matrix.csv`; (6) verified the three workbooks (six-run comparison sheet, learning-curve charts including the test MCEL series, A-Z confusion headers with live total/PGC formulas) and REPORT.pdf (derivation and experiment analysis); and (7) created the final submission ZIP from the V2 directory. This section was added by that session so the record remains complete and truthful.

### 7. 16,000-row protocol regeneration (WorkBuddy session, 7 September 2026, V3)

User prompt: quoting the V2 delivery note that the v5 starter's 12,800/3,200/4,000 split differs from the assignment PDF's 16,000-training wording, the student replied "请按作业要求重新写" (redo it according to the assignment requirements).

Assistant response summary: this session rebuilt the submission under the assignment PDF's 16,000-row training requirement using a two-stage protocol. (1) `training_set.csv` was expanded to 16,000 rows by concatenating the v5 starter's training file (12,800 rows) and validation file (3,200 rows) in original order; the first 12,800 rows remain byte-identical to the starter training file, and `validation_set.csv` / `test_set.csv` are unchanged. (2) `experiments.cpp` was modified so stage 1 trains six candidates on the first 12,800 rows and selects by validation MCEL (reproducing the V2 selection byte-identically - all six run logs matched the V2 logs exactly), while stage 2 retrains the winning configuration (ReLU 16-64-32-26, learning rate 0.03) from scratch on all 16,000 rows for the selected 143 epochs. (3) The final model's outputs - `best_weights.txt`, `final_training_log.csv`, `confusion_matrix.csv`, `best_result.txt`, `training_log.txt` - were regenerated: training 15,446/16,000 (96.5375%), validation-file rows in-sample 3,092/3,200, test 3,725/4,000 (93.125%, MCEL 0.22114756). (4) `performance_and_confusion.xlsx` and `bestNetwork_25021827.xlsx` were rebuilt with the final-model curves (epochs 0-143, training and test), the new confusion matrix and updated notes; README.md, REPORT.pdf and VERIFICATION.md were rewritten for the new protocol. (5) The rebuilt engine compiled without warnings, passed `--self-test`, and two independent verifiers (a separate C++ loader and a NumPy implementation that parses `best_weights.txt` directly) reproduced all final metrics within 1e-12 with zero confusion mismatches. (6) The GUI smoke-test dataset check was updated to 16,000 training rows. This section was added by that session so the record remains complete and truthful.

## Verification and provenance

The assistant performed the checks listed in VERIFICATION.md. This does not establish that the student has understood or independently verified the code. The submitted code and documents remain AI-assisted work and should be explained and disclosed as such.

Algorithm references are cited in REPORT.pdf. They were used as mathematical references for initialization, not as copied source implementations. The starting point was the instructor-supplied v5 starter project and datasets. The original class/workflow structure was adapted and substantially rewritten for multiple layers and softmax.
