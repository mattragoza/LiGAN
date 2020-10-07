import re
import pandas as pd

csv_file = 'problem.csv.fixed'

try:
    df = pd.read_csv(csv_file, sep=' ')
    n_fields_expected = len(df.columns)
    print('Parsed successfully')
    parse_ok = True
except pd.errors.ParserError as e:
    print(e)
    m = re.match(r'Error tokenizing data. C error: Expected (\d+) fields in line (\d+), saw (\d+)', str(e))
    n_fields_expected = int(m.group(1))
    n_fields_parsed = int(m.group(3))
    row_num = int(m.group(2))
    parse_ok = False

with open(csv_file) as f:
    buf = f.read()

line_nums = [1]
rows = []
curr_line_num = 1
curr_row = []
curr_field = ''
in_quote = False

for c in buf:
    if c == '\n':
        curr_line_num += 1
    if in_quote:
        if c == '"':
            in_quote = False
            curr_field = '"..."'
        else:
            curr_field += c
    else:
        if c == '\n':
            curr_row.append(curr_field)
            rows.append(curr_row)
            line_nums.append(curr_line_num)
            curr_row = []
            curr_field = ''
        elif c == ' ':
            curr_row.append(curr_field)
            curr_field = ''
        elif c == '"':
            in_quote = True
        else:
            curr_field += c

assert not in_quote

if c != '\n':
    curr_row.append(curr_field)
    rows.append(curr_row)
    line_nums.append(curr_line_num)

n_problems = 0
for i, row in enumerate(rows):
    if len(row) != n_fields_expected:
        print('Row {} has {} fields ({} {})'.format(i+1, len(row), row[0], row[1]))
        n_problems += 1
        if parse_ok:
            row_num = i+1
            parse_ok = False
print('{} problems left'.format(n_problems))

# 3777602_4 9

n_context = 2
context_offsets = range(-n_context-1, n_context)

columns = rows[0]
max_n_fields = len(columns)
for i in context_offsets:
    n_fields = len(rows[row_num+i])
    if n_fields > max_n_fields:
        max_n_fields = n_fields


def abbrev(s, w):
    s = s.rjust(w)
    if len(s) > w:
        s = s[:w-3] + '...'
    return s

headers = ['', 'row (line)'] + [str(row_num+j+1) + ' (' + str(line_nums[row_num+j]) + ')' for j in context_offsets]
col_widths = [8, 30] + [20 for i in context_offsets]
print(' '.join([abbrev(s, w) for s,w in zip(headers, col_widths)]))

for i in range(max_n_fields):
    values = [str(i+1)]
    try:
        column = columns[i]
    except IndexError:
        column = ''
    values.append(column)

    for j in context_offsets:
        row = rows[row_num+j]
        try:
            field = row[i] or 'nan'
        except IndexError:
            field = ''
        values.append(field)

    print(' '.join([abbrev(s, w) for s,w in zip(values, col_widths)]))

