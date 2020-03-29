class PDTBInstance:
    def __init__(self, guid, text_a, text_b=None, label=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
    def init_from_pipe_instance(self, pipe_instance):
        self.text_a = pipe_instance.arg1
        self.text_b = pipe_instance.arg2
        self.label = pipe_instance.label

class PipeInstance:
    def __init__(self, columns, text):
        self.type = columns[0]
        self.connective = columns[2]
        self.connective_pos = columns[1].split(';')

        self.conn1 = columns[7] # Explicit or 1st Implicit Connective
        self.conn2 = columns[10] # 2nd Implicit Connective
        self.conn1_sem1, self.conn1_sem2 = columns[8], columns[9]
        self.conn2_sem1, self.conn2_sem2 = columns[11], columns[12]
        self.sems = list({self.conn1_sem1, self.conn1_sem2, self.conn2_sem1, self.conn1_sem2})
        self.sems.remove('')
        self.label = get_label(self.sems)
        
        self.arg1 = None
        self.arg2 = None
        self.arg1_pos = columns[14].split(';')
        self.arg2_pos = columns[20].split(';')
        paired_connectives = {"if then", "either or", "neither nor","not only but also", "not only but", "not only also", "not just but", "both and", "on the one hand on the other", "not so much as"}
        arg1_connective = {'not only'}
        if self.type == 'Explicit':
            conn_pos = ret_pos_set(self.connective_pos)
            if self.conn1 in paired_connectives:
                arg1_pos = ret_pos_set(self.arg1_pos) + conn_pos[0:2]
                arg2_pos = ret_pos_set(self.arg2_pos) + conn_pos[2:4]
                arg1_txt, arg2_txt = self._pos_to_txt(text, arg1_pos, arg2_pos)
                # Check if two args have overlap
                if not self._if_overlap(arg1_txt, arg2_txt):
                    self.arg1 = arg1_txt
                    self.arg2 = arg2_txt
            elif self.conn1 in arg1_connective:
                arg1_pos = ret_pos_set(self.arg1_pos) + conn_pos
                arg2_pos = ret_pos_set(self.arg2_pos)
                arg1_txt, arg2_txt = self._pos_to_txt(text, arg1_pos, arg2_pos)
                if not self._if_overlap(arg1_txt, arg2_txt):
                    self.arg1 = arg1_txt
                    self.arg2 = arg2_txt
                    
            else:
                arg1_pos = ret_pos_set(self.arg1_pos)
                arg2_pos = ret_pos_set(self.arg2_pos) + conn_pos
                arg1_txt, arg2_txt = self._pos_to_txt(text, arg1_pos, arg2_pos)
                if not self._if_overlap(arg1_txt, arg2_txt):
                    self.arg1 = arg1_txt
                    self.arg2 = arg2_txt
        else:
            arg1_pos = ret_pos_set(self.arg1_pos)
            arg2_pos = ret_pos_set(self.arg2_pos)
            arg1_txt, arg2_txt = self._pos_to_txt(text, arg1_pos, arg2_pos)
            if not self._if_overlap(arg1_txt, arg2_txt):
                self.arg1 = arg1_txt
                self.arg2 = arg2_txt

    def _pos_to_txt(self, text, _arg1, _arg2):
        _arg1.sort()
        _arg2.sort()
        return text[_arg1[0] : _arg1[-1]], text[_arg2[0] : _arg2[-1]]
    
    def _if_overlap(self, _arg1, _arg2):
        if _arg1.find(_arg2) != -1:
            return True
        if _arg2.find(_arg1) != -1:
            return True
        return False

    def print_inst(self):
        print('-----------------------------------------')
        print("Type:", self.type)
        print("Conn1:", self.conn1)
        print("Arg1 Pos:", self.arg1_pos)
        print("Arg2 Pos:", self.arg2_pos)
        print("Arg1:", self.arg1)
        print("Arg2:", self.arg2)
        print("Conn1 Sem1:", self.conn1_sem1)
        print("Sems:", self.sems)
        print("Label:", self.label)
        print('-----------------------------------------')

def ret_pos_set(pos):
    ret = []
    for p in pos:
        if p == "":
            continue
        tmp = p.split('..')
        ret.append(int(tmp[0]))
        ret.append(int(tmp[1]))
    return ret

def get_label(sems):
    for sem in sems:
        sem_items = sem.split('.')
        if sem_items[0] == "Contingency":
            # "Contingency.Cause.Reason" : Arg1 -> effect / Arg2 -> cause : "1"
            # "Contingency.Cause.Result" : Arg1 -> cause / Arg2 -> effect : "2"
            sem_items[1] = sem_items[1].split('+')[0]
            if sem_items[1] == "Cause":
                sem_items[2] = sem_items[2].split('+')[0]
                if sem_items[2] == "Reason":
                    return("1")
                elif sem_items[2] == "Result":
                    return("2")
    return("0")

def load_pipe_file(rpath, gpath, types=None):
    with open(rpath, 'r', encoding='ISO-8859-1') as fin:
        text = fin.read()
    with open(gpath, 'r', encoding='ISO-8859-1') as fin:
        lines = fin.readlines()
    instances = []
    for line in lines:
        columns = line.split('|')
        instance = PipeInstance(columns, text)
        if types is not None and instance.type not in types:
            continue
        if instance.arg1 is not None and instance.arg2 is not None:
            instances.append(instance)
    return instances