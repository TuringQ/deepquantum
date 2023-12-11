from collections import defaultdict

import numpy as np
import svgwrite

from .gate import PhaseShift, BeamSplitter, UAnyGate


class DrawCircuit():
    """
    draw photonic circuit
    """
    def __init__(self, circuit_name, circuit_nmode, circuit_operators, circuit_depth) -> None:
        if circuit_name is None:
            circuit_name = 'circuit'
        # self.circuit = qumodeCircuit
        n_mode = circuit_nmode
        name = circuit_name + '.svg'
        draw_circuit = svgwrite.Drawing(name, profile='full')
        # size = n_mode/4*100   # the size of SVG circuit
        # size = 5*100
        # wid = max(1, max(circuit_depth))
        # wid = (90*20+40)/100
        # draw_circuit['width'] = f'{3 * wid}cm'
        draw_circuit['height'] = f'{10/11 * n_mode}cm'
        self.draw_ = draw_circuit
        self.n_mode = n_mode
        self.name = name
        self.ops = circuit_operators
        print(circuit_depth)

    def draw(self):
        order_dic = defaultdict(list) # 当key不存在时对应的value是[]
        n_mode = self.n_mode
        ps_num = ''
        depth = [0] * n_mode # record the depth of each mode
        for _, op in enumerate(self.ops):
            op_wires = op.wires
            if isinstance(op, BeamSplitter):
                theta = op.theta.detach().numpy()
                phi = op.phi.detach().numpy()
                order = max(depth[op_wires[0]], depth[op_wires[1]])
                self.draw_bs(order, op_wires, theta, phi)
                order_dic[order] = order_dic[order] + op_wires
                for i in op_wires:
                    depth[i] = depth[i] + 1
                bs_depth = [depth[op_wires[0]], depth[op_wires[1]]][:]
                depth[op_wires[0]] = max(bs_depth)           ## BS 经过后相同线路深度
                depth[op_wires[1]] = max(bs_depth)
            if isinstance(op, PhaseShift):
                theta = op.theta.detach().numpy()
                order = depth[op_wires[0]]
                self.draw_ps(order, op.wires, theta)
                order_dic[order] = order_dic[order] + op_wires
                for i in op.wires:
                    depth[i] = depth[i]+1
            if isinstance(op, UAnyGate): # need check?
                order = max(depth[op_wires[0] : op_wires[-1]+1])
                self.draw_any(order, op_wires)
                order_dic[order] = order_dic[order] + op_wires
                for i in op_wires:
                    depth[i] = order + 1
        for key in order_dic.keys():
            op_line = order_dic[key]  ## here lines represnet for no operation
            line_wires = [i for i in range(n_mode) if i not in op_line]
            if len(line_wires) > 0:
                self.draw_lines(key, line_wires)
        self.draw_mode_num()   ## mode draw numbers
        self.order_dic = order_dic
        self.depth = depth
        self.ps_num = ps_num
        print(self.depth)

    def save(self, filename):
        """
        save the circuit as svg
        """
        self.draw_.saveas(filename)

    def draw_mode_num(self):
        n_mode =  self.n_mode
        for i in range(n_mode):
            self.draw_.add(self.draw_.text(str(i), insert=(25, i*30+30), font_size=12))

    def draw_bs(self, order, wires, theta, phi):
        """
        draw beamsplitter
        """
        x = 90 * order + 40
        y_up = wires[0]
        # y_down = wires[1]
        self.draw_.add(self.draw_.polyline(points=[(x, y_up*30+30), (x+30, y_up*30+30),
                                                   (x+60, y_up*30+30+30), (x+90, y_up*30+30+30)],
                                           fill='none', stroke='black', stroke_width=2))
        self.draw_.add(self.draw_.polyline(points=[(x, y_up*30+30+30), (x+30, y_up*30+30+30),
                                                   (x+60, y_up*30+30), (x+90, y_up*30+30)],
                                           fill='none', stroke='black', stroke_width=2))
        self.draw_.add(self.draw_.text('BS', insert=(x+40, y_up*30+30), font_size=9))
        self.draw_.add(self.draw_.text('θ ='+ str(np.round(theta,3)),
                                       insert=(x+55, y_up*30+30+20-6),
                                       font_size=7))
        self.draw_.add(self.draw_.text('ϕ ='+ str(np.round(phi,3)),
                                       insert=(x+55, y_up*30+30+26-6),
                                       font_size=7))

    def draw_ps(self, order, wires, theta):
        """
        draw phaseshift
        """
        x = 90 * order + 40
        y_up = wires[0]
        # y_down = wires[1]
        self.draw_.add(self.draw_.polyline(points=[(x, y_up*30+30), (x+90, y_up*30+30)],
                                           fill='none', stroke='black', stroke_width=2))
        self.draw_.add(self.draw_.rect(insert=(x+45, y_up*30+25), size=(6,12), rx=0, ry=0,
                                       fill='none', stroke='black', stroke_width=1.5))
        self.draw_.add(self.draw_.text('PS', insert=((x+45), y_up*30+20), font_size=9))
        self.draw_.add(self.draw_.text('λ ='+str(np.round(theta,3 )), insert=(x+60, y_up*30+20), font_size=7))

    def draw_any(self, order, wires):
        """
        draw arbitrary unitary gate
        """
        x = 90 * order + 40
        y_up = wires[0]
        h = (int(len(wires)) - 1) * 30 + 20
        for k in wires:
            self.draw_.add(self.draw_.polyline(points=[(x, k*30+30),(x+20, k*30+30)],
                                               fill='none', stroke='black', stroke_width=2))

            self.draw_.add(self.draw_.polyline(points=[(x+70, k*30+30),(x+90, k*30+30)],
                                               fill='none', stroke='black', stroke_width=2))

        self.draw_.add(self.draw_.rect(insert=(x+20, y_up*30+20), size=(50, h), rx=0, ry=0,
                                       fill='none', stroke='black', stroke_width=2))
        self.draw_.add(self.draw_.text('U', insert=((x+41), y_up*30+20+h/2), font_size=10))

    def draw_lines(self, order, wires):
        """
        for acting nothing
        """
        x = 90 * order + 40
        for k in wires:
            self.draw_.add(self.draw_.polyline(points=[(x, k*30+30),(x+90, k*30+30)],
                                               fill='none', stroke='black', stroke_width=2))
