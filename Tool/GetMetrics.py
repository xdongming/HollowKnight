def get_se(name, loss):
    loss = float(loss)
    filename_act = './result/act_se.txt'
    filename_move = './result/move_se.txt'
    if name == 'act':
        with open(filename_act, 'a') as file_object:
            file_object.write(str(loss)+' ')
    elif name == 'move':
        with open(filename_move, 'a') as file_object:
            file_object.write(str(loss)+' ')

def get_q_value(name, value):
    filename_act = './result/act_q.txt'
    filename_move = './result/move_q.txt'
    if name == 'act':
        with open(filename_act, 'a') as file_object:
            file_object.write(str(value)+' ')
    elif name == 'move':
        with open(filename_move, 'a') as file_object:
            file_object.write(str(value)+' ')

def get_pass_count(number):
    filename = './result/pass_count.txt'
    f = open(filename, "r", encoding="utf-8")
    data = f.read()
    if not data:
        f.close()
        count = str(number)
        with open(filename, 'a') as file_object:
            file_object.write(count+' ')
    else:
        listA = data.split(' ')
        f.close()
        count = str(int(listA[-2]) + number)
        with open(filename, 'a') as file_object:
            file_object.write(count+' ')
