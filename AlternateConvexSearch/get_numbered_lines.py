import sys

if len(sys.argv) != 3:
    print "These args: "+", ".join(sys.argv)
    print "Correct usage: python get_numbered_lines.py [numbers file] [lines file]"
    print "Prints all numbered lines to the screen; redirect to file by appending"
    print "> [output file] to the command"
    sys.exit(1)

numbers_filename = sys.argv[1]
other_filename = sys.argv[2]

numbers_file = open(numbers_filename)
numbers = [0] # always copy top line
for line in numbers_file.read():
    try:
        numbers.append(int(line))
    except Exception:
        1;
        # print "error on line: "+line
numbers_file.close()

other_file = open(other_filename)
line_num = 1
lines = [other_file.readline()];
line = other_file.readline()
while line:
    if line_num in numbers:
        lines.append(line)
    line_num += 1
    line = other_file.readline()
other_file.close()
    
print "".join(lines)