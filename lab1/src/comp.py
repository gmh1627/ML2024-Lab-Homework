import difflib

def compare_files(file1, file2):
    with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
        diff = difflib.unified_diff(
            f1.readlines(),
            f2.readlines(),
            fromfile=file1,
            tofile=file2,
            lineterm=''
        )
        for line in diff:
            print(line)

# 使用示例
compare_files('submission.py', 'submission(t).py')