"""
Draw quantum circuit 
"""
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import svgwrite

from .gate import PhaseShift, BeamSplitter, UAnyGate


class DrawCircuit():
    """
    draw photonic circuit
    """
    def __init__(self, circuit_name, circuit_nmode, circuit_operators) -> None:
        if circuit_name is None:
            circuit_name = 'circuit'
        n_mode = circuit_nmode
        name = circuit_name + '.svg'
        draw_circuit = svgwrite.Drawing(name, profile='full')
        draw_circuit['height'] = f'{10/11 * n_mode}cm'
        self.draw_ = draw_circuit
        self.n_mode = n_mode
        self.name = name
        self.ops = circuit_operators

    def draw(self):
        order_dic = defaultdict(list) # 当key不存在时对应的value是[]
        n_mode = self.n_mode
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
            elif isinstance(op, PhaseShift):
                theta = op.theta.detach().numpy()
                order = depth[op_wires[0]]
                self.draw_ps(order, op.wires, theta)
                order_dic[order] = order_dic[order] + op_wires
                for i in op.wires:
                    depth[i] = depth[i]+1
            elif isinstance(op, UAnyGate): # need check?
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
        wid = 3 * (90 * (max(self.depth)) + 40) / 100
        self.draw_['width'] = f'{wid}cm'

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

class Graph_Mzi():
    """
    for plotting mzi clements structure
    n_mode: int
    mzi_info: dictionary for mzi parameters, result for decomse function
    cl: color for plotting
    fs: fontsize
    type: the way for clements decomposition, cssr or cssl
    """
    def __init__(self, n_mode, mzi_info, cl='dodgerblue', fs=30, type_='cssr'):
        self.n_mode = n_mode
        self.type = type_
        self.mzi_info = mzi_info
        self.color = cl
        self.fontsize =fs
        self.wid = 0.1
        self.height = 0.08
        self.axis_off = 'off'
        self.phase_angle = self.mzi_info['phase_angle'] # for phase shifter
        self.dic_mzi = self.sort_mzi() # for mzi parameters in the same array
        self.ps_position = self.ps_pos()

    def plotting_clements(self):
        if self.type == 'cssr':
            assert(self.n_mode%2 == 0), 'plotting only valid for even modes'
            self.plotting_clements_1()
        if self.type == 'cssl':
            self.plotting_clements_2()

    def plotting_clements_1(self):
        """
        plotting CSSR, order: left to right
        """
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8*3,5*3)
        # plt.rcParams['figure.figsize'] = (8*3,5.0*3)
        coords1 = []
        coords2 = []
        n_mode = self.n_mode
        phase_angle = self.phase_angle
        fs = self.fontsize
        cl = self.color
        wid = self.wid
        height = self.height
        for i in range(n_mode):
            plt.annotate('',xy=(-0.1,1-0.25*i),
                        xytext=(-0.5,1-0.25*i),
                        arrowprops={'arrowstyle': '-|>', 'lw':5},
                        va = 'center',)
            plt.text(-0.8, 1-0.25*i, f'{i}', fontsize = fs )
            plt.plot([0, 1.2], [1-0.25*i,1-0.25*i], color = cl)
            plt.text( 3.2*(n_mode/2-1)+2.2+2.1, 1-0.25*i+0.05, f'{phase_angle[i]:.2f}', fontsize=fs-8 )  # phase angle
            ax.add_patch(
            patches.Rectangle(
                (3.2*(n_mode/2-1)+2.2+2.1, 1-0.25*i-0.05),
                wid,
                height,
                edgecolor = 'green',
                facecolor = 'green',
                fill=True
                             ) )  ## for PS
            if n_mode%2==1:
                plt.plot([2.2+3.2*(int((n_mode+1)/2)-1),
                          3.2*int((n_mode+1)/2-1)+2.2+2.2],
                          [1-0.25*i,1-0.25*i],
                          color = cl)
        if n_mode%2==0:   # for even mode
            for i in range(int(n_mode/2)):
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1,1], color = cl)
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1-0.25*(n_mode-1), 1-0.25*(n_mode-1) ], color = cl)
                for j in range(n_mode):
                    plt.plot([1.5+3.2*i, 3.2*i+1.9], [1-0.25*j,1-0.25*j], color = cl)
                    coords1.append( [1.5+3.2*i, 3.2*i+1.9, 1-0.25*j,1-0.25*j])
                    if 0<j<n_mode-1:
                        plt.plot([3.1+3.2*i, 3.2*i+3.5], [1-0.25*j,1-0.25*j], color = cl)
                        coords2.append([3.1+3.2*i, 3.2*i+3.5, 1-0.25*j,1-0.25*j])
                        plt.plot([2.2+3.2*i, 3.2*i+2.8], [1-0.25*j,1-0.25*j], color = cl)
                        plt.plot([3.8+3.2*i, 3.2*i+4.4], [1-0.25*j,1-0.25*j], color = cl)
        if n_mode%2==1:  # for odd mode
            for i in range(int((n_mode+1)/2)):
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1,1], color = cl)
            #     plt.plot([1.2+3.2*i, 3.2*i+2.2+2.2], [1-0.25*(n_mode-1), 1-0.25*(n_mode-1) ], color = cl)
                for j in range(n_mode):
                    if j< n_mode-1: # remove last line
                        plt.plot([1.5+3.2*i, 3.2*i+1.9], [1-0.25*j,1-0.25*j], color = cl)
                        coords1.append( [1.5+3.2*i, 3.2*i+1.9, 1-0.25*j,1-0.25*j])
                    if j >= n_mode-1:
                        plt.plot([1.2+3.2*i, 3.2*i+2.2], [1-0.25*j,1-0.25*j], color = cl)
                    if  i< int((n_mode+1)/2)-1 and 0<j<n_mode: # remove the last column
                        plt.plot([3.1+3.2*i, 3.2*i+3.5], [1-0.25*j,1-0.25*j], color = cl)
                        coords2.append([3.1+3.2*i, 3.2*i+3.5, 1-0.25*j,1-0.25*j])
                        plt.plot([2.2+3.2*i, 3.2*i+2.8], [1-0.25*j,1-0.25*j], color = cl)
                        plt.plot([3.8+3.2*i, 3.2*i+4.4], [1-0.25*j,1-0.25*j], color = cl)
        # connecting lines i, i+1
        for i  in range(len(coords1)):
            if i%2==0:
                self.connect1(coords1[i], ax, a=-0.5-0.4, c=0.7-0.7)
            if i%2==1:
                self.connect2(coords1[i])
        for i  in range(len(coords2)):
            if i%2==0:
                self.connect1(coords2[i], ax, a=-0.5-0.4, c=0.7-0.7)
            if i%2==1:
                self.connect2(coords2[i])
        # plotting paras
        self.plot_paras_1(self.dic_mzi, fs=self.fontsize-8)
        plt.axis(self.axis_off)
        # if self.axis_off:
        #     plt.axis('off')
        plt.show()

    def plotting_clements_2(self):
        """
        plotting CSSL, order: right to left
        """
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8*3,5*3)
        # plt.rcParams['figure.figsize'] = (8*3,5.0*3)
        coords1 = []
        coords2 = []
        n_mode = self.n_mode
        phase_angle = self.phase_angle
        fs = self.fontsize
        cl = self.color
        wid = self.wid
        height = self.height
        for i in range(n_mode):
            plt.annotate('',xy=(-0.1,1-0.25*i),
                         xytext=(-0.5,1-0.25*i),
                         arrowprops={'arrowstyle': '-|>', 'lw':5},
                         va='center',)
            plt.text(-0.8, 1-0.25*i, f'{i}', fontsize = fs )
            plt.plot([0, 1.2], [1-0.25*i,1-0.25*i], color = cl)
            plt.text( 0.4,1-0.25*i+0.05, f'{phase_angle[i]:.2f}', fontsize=fs-8 )  # phase angle
            ax.add_patch(
            patches.Rectangle(
                (0.5,1-0.25*i-0.05),
                wid,
                height,
                edgecolor = 'blue',
                facecolor = 'blue',
                fill=True
                             ) )
            if n_mode%2==1:
                plt.plot([2.2+3.2*(int((n_mode+1)/2)-1),
                          3.2*int((n_mode+1)/2-1)+2.2+2.2],
                          [1-0.25*i,1-0.25*i],
                          color = cl)
        if n_mode%2==0:   # for even mode
            for i in range(int(n_mode/2)):
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1,1], color = cl)
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1-0.25*(n_mode-1), 1-0.25*(n_mode-1) ], color = cl)
                for j in range(n_mode):
                    plt.plot([1.5+3.2*i, 3.2*i+1.9], [1-0.25*j,1-0.25*j], color = cl)
                    coords1.append([1.5+3.2*i, 3.2*i+1.9, 1-0.25*j,1-0.25*j])
                    if 0<j<n_mode-1:
                        plt.plot([3.1+3.2*i, 3.2*i+3.5], [1-0.25*j,1-0.25*j], color = cl)
                        coords2.append([3.1+3.2*i, 3.2*i+3.5, 1-0.25*j,1-0.25*j])
                        plt.plot([2.2+3.2*i, 3.2*i+2.8], [1-0.25*j,1-0.25*j], color = cl)
                        plt.plot([3.8+3.2*i, 3.2*i+4.4], [1-0.25*j,1-0.25*j], color = cl)
        if n_mode%2==1:  # for odd mode
            for i in range(int((n_mode+1)/2)):
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1,1], color = cl)
            #     plt.plot([1.2+3.2*i, 3.2*i+2.2+2.2], [1-0.25*(n_mode-1), 1-0.25*(n_mode-1) ], color = cl)
                for j in range(n_mode):
                    if j< n_mode-1: # remove last line
                        plt.plot([1.5+3.2*i, 3.2*i+1.9], [1-0.25*j,1-0.25*j], color = cl)
                        coords1.append( [1.5+3.2*i, 3.2*i+1.9, 1-0.25*j,1-0.25*j])
                    if j >= n_mode-1:
                        plt.plot([1.2+3.2*i, 3.2*i+2.2], [1-0.25*j,1-0.25*j], color = cl)
                    if  i< int((n_mode+1)/2)-1 and 0<j<n_mode: # remove the last column
                        plt.plot([3.1+3.2*i, 3.2*i+3.5], [1-0.25*j,1-0.25*j], color = cl)
                        coords2.append([3.1+3.2*i, 3.2*i+3.5, 1-0.25*j,1-0.25*j])
                        plt.plot([2.2+3.2*i, 3.2*i+2.8], [1-0.25*j,1-0.25*j], color = cl)
                        plt.plot([3.8+3.2*i, 3.2*i+4.4], [1-0.25*j,1-0.25*j], color = cl)
        # connecting lines i, i+1
        for i  in range(len(coords1)):
            if i%2==0:
                self.connect1(coords1[i], ax)
            if i%2==1:
                self.connect2(coords1[i])
        for i  in range(len(coords2)):
            if i%2==0:
                self.connect1(coords2[i], ax)
            if i%2==1:
                self.connect2(coords2[i])
        # plotting paras
        self.plot_paras(self.dic_mzi, self.n_mode, fs=self.fontsize-8)
        plt.axis(self.axis_off)
        # if self.axis_off:
        #     plt.axis('off')
        plt.show()

    def sort_mzi(self):
        """
        sort mzi parameters in the same array for plotting
        """
        dic_mzi = defaultdict( list) #当key不存在时对应的value是[]
        mzi_list = self.mzi_info['MZI_list']
        for i in mzi_list:
            dic_mzi[tuple(i[0:2])].append(i[2:])
        return dic_mzi

    def ps_pos(self):
        """
        label the position of each phaseshifter for cssr case
        """
        if self.type == 'cssr':
            dic_pos = { }
            nmode = self.n_mode
            phase_angle = self.phase_angle
            dic_ =self.dic_mzi
            for mode in range(nmode):
                pair = (mode, mode+1)
                value = dic_[pair]
                value = np.array(value).flatten()
                for k in range(len(value)):
                    dic_pos[(mode, k)] = np.round(value[k], 4)
                if mode == nmode -1:
                    dic_pos[(mode, 0)] = np.round(phase_angle[mode], 4)
                else:
                    dic_pos[(mode, k+1)] = np.round(phase_angle[mode], 4)
            return dic_pos
        else:
            return None

    @staticmethod
    def connect1(coordinate, ax, cl='dodgerblue', wid=0.1, height=0.08, a=-0.05, b=-0.05, c=0.7, d=-0.05):
        """
        connect odd column
        """
        x0, x1, y0, y1 = coordinate
    #     print(x0,x1,y0,y1)
        plt.plot([x0, x0-0.3],[y0, y0-0.25], color = cl)
        plt.plot([x1, x1+0.3],[y1, y1-0.25], color = cl)
        ax.add_patch(patches.Rectangle(
            ((x0+x1)/2 + a, y0 + b),
            wid,
            height,
            edgecolor = 'blue',
            facecolor = 'blue',
            fill=True
        ) )
        ax.add_patch(patches.Rectangle(
            ((x0+x1)/2 + c, y0 + d),
            wid,
            height,
            edgecolor = 'blue',
            facecolor = 'blue',
            fill=True
        ) )

    @staticmethod
    def connect2(coordinate, cl='dodgerblue'):
        """
        connect even column
        """
        x0, x1, y0, y1 = coordinate
        plt.plot([x0, x0-0.3],[y0, y0+0.25], color = cl)
        plt.plot([x1, x1+0.3],[y1, y1+0.25], color = cl)

    @staticmethod
    def plot_paras(sort_mzi_dic, n_mode, fs=20):
        """
        plotting mzi_paras, for CSSL
        """
        for i in sort_mzi_dic.keys():
            if i[0]%2 == 0: # 0, 2, 4, 6..
                temp_values = sort_mzi_dic[i]
                len_ = len(temp_values)
                for j in range(len_):
                    plt.text(8.6-3.2*j+3.2*((n_mode-6)//2+n_mode%2),
                             1-0.25*i[0]+0.05,
                             f'{temp_values[j][0]:.2f}',
                             fontsize=fs)
                    plt.text(7.8-3.2*j+3.2*((n_mode-6)//2+n_mode%2),
                             1-0.25*i[0]+0.05,
                             f'{temp_values[j][1]:.2f}',
                             fontsize=fs)
            if i[0]%2 ==1: # 1, 3..
                temp_values = sort_mzi_dic[i]
                len_ = len(temp_values)
                for j in range(len_):
                    plt.text(8.6-3.2*j+1.6+3.2*((n_mode-6)//2),
                             1-0.25*i[0]+0.05,
                             f'{temp_values[j][0]:.2f}',
                             fontsize=fs)
                    plt.text(7.8-3.2*j+1.6+3.2*((n_mode-6)//2),
                             1-0.25*i[0]+0.05,
                             f'{temp_values[j][1]:.2f}',
                             fontsize=fs)

    @staticmethod
    def plot_paras_1(sort_mzi_dic, fs=20):
        """
        plotting mzi_paras, CSSR
        """
        for i in sort_mzi_dic.keys():
            if i[0]%2 == 0: # 0, 2, 4, 6..
                temp_values = sort_mzi_dic[i]
                len_ = len(temp_values)
                for j in range(len_):
                    plt.text(3.2*j+0.6, 1-0.25*i[0]+0.05, f'{temp_values[j][0]:.2f}', fontsize=fs)
                    plt.text(3.2*j+0.6+0.9, 1-0.25*i[0]+0.05, f'{temp_values[j][1]:.2f}', fontsize=fs)
            if i[0]%2 ==1: # 1, 3..
                temp_values = sort_mzi_dic[i]
                len_ = len(temp_values)
                for j in range(len_):
                    plt.text(3.2*j+0.6+1.6, 1-0.25*i[0]+0.05, f'{temp_values[j][0]:.2f}', fontsize=fs)
                    plt.text(3.2*j+0.6+2.4, 1-0.25*i[0]+0.05, f'{temp_values[j][1]:.2f}', fontsize=fs)
