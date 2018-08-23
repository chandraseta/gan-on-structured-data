with open('gan-bce.log', 'r') as file:
    line = file.readline()

# Process lines and output to file
is_even_dot = False
is_first = True
new_line_idx = 0
with open('gan-bce-formatted.log', 'w') as output_file:
    for idx, c in enumerate(line):
        if c == '.':
            if not is_first:
                if is_even_dot:
                    sub = line[new_line_idx:idx-1]
                    new_line_idx = idx-1
                    output_file.write('{}\n'.format(sub))

                    is_even_dot = False
                else:
                    is_even_dot = True
        elif c == ',':
            is_first = False
