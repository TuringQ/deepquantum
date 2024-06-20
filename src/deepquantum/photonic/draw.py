"""
Draw quantum circuit.
"""

from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import svgwrite
from matplotlib import patches
from torch import nn

from .gate import PhaseShift, BeamSplitter, MZI, BeamSplitterSingle, UAnyGate, Squeezing, Displacement

info_dic = {'PS': ['teal', 0],
            'S': ['royalblue', 3],
            'D': ['green', 3],
            'U': ['cadetblue', 0]
            }

class DrawCircuit():
    """Draw the photonic quantum circuit.

    Args:
        circuit_name (str): The name of the circuit.
        circuit_nmode (int): The number of modes in the circuit.
        circuit_operators (nn.Sequential): The operators of the circuit.
    """
    def __init__(self, circuit_name: str, circuit_nmode: int, circuit_operators: nn.Sequential) -> None:
        if circuit_name is None:
            circuit_name = 'circuit'
        nmode = circuit_nmode
        name = circuit_name + '.svg'
        self.draw_ = svgwrite.Drawing(name, profile='full')
        self.draw_['height'] = f'{10/11 * nmode}cm'
        self.nmode = nmode
        self.name = name
        self.ops = circuit_operators

    def draw(self):
        """Draw circuit."""
        order_dic = defaultdict(list) # 当key不存在时对应的value是[]
        nmode = self.nmode
        depth = [0] * nmode # record the depth of each mode
        for op in self.ops:
            if isinstance(op, BeamSplitter):
                if isinstance(op, MZI):
                    if op.phi_first:
                        name = 'MZI-PT'
                    else:
                        name = 'MZI-TP'
                elif isinstance(op, BeamSplitterSingle):
                    name = 'BS-' + op.convention.upper()
                else:
                    name = 'BS'
                theta = op.theta.item()
                try:
                    phi = op.phi.item()
                except:
                    phi = None
                order = max(depth[op.wires[0]], depth[op.wires[1]])
                self.draw_bs(name, order, op.wires, theta, phi)
                order_dic[order] = order_dic[order] + op.wires
                for i in op.wires:
                    depth[i] = depth[i] + 1
                bs_depth = [depth[op.wires[0]], depth[op.wires[1]]][:]
                depth[op.wires[0]] = max(bs_depth)           ## BS 经过后相同线路深度
                depth[op.wires[1]] = max(bs_depth)
            elif isinstance(op, PhaseShift):
                name_ = 'PS'
                theta = op.theta.item()
                order = depth[op.wires[0]]
                self.draw_ps(order, op.wires, theta, name_)
                order_dic[order] = order_dic[order] + op.wires
                for i in op.wires:
                    depth[i] = depth[i]+1
            elif isinstance(op, UAnyGate): # need check?
                order = max(depth[op.wires[0] : op.wires[-1]+1])
                name_ = 'U'
                self.draw_any(order, op.wires, name_)
                order_dic[order] = order_dic[order] + op.wires
                for i in op.wires:
                    depth[i] = order + 1
            elif isinstance(op, (Squeezing, Displacement)):
                r = op.r.item()
                theta = op.theta.item()
                order = depth[op.wires[0]]
                if isinstance(op, Squeezing):
                    name_ = 'S'
                else:
                    name_ = 'D'
                self.draw_sq(order, op.wires, r, theta, name_)
                order_dic[order] = order_dic[order] + op.wires
                for i in op.wires:
                    depth[i] = depth[i]+1

        for key, value in order_dic.items():
            op_line = value  ## here lines represent for no operation
            line_wires = [i for i in range(nmode) if i not in op_line]
            if len(line_wires) > 0:
                self.draw_lines(key, line_wires)
        self.draw_mode_num()   ## mode draw numbers
        self.order_dic = order_dic
        self.depth = depth
        wid = 3 * (90 * (max(self.depth)) + 40) / 100
        self.draw_['width'] = f'{wid}cm'

    def save(self, filename):
        """
        Save the circuit as svg.
        """
        self.draw_.saveas(filename)

    def draw_mode_num(self):
        nmode =  self.nmode
        for i in range(nmode):
            self.draw_.add(self.draw_.text(str(i), insert=(25, i*30+30), font_size=12))

    def draw_bs(self, name, order, wires, theta, phi = None):
        """
        Draw beamsplitter.
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
        self.draw_.add(self.draw_.text(name, insert=(x+40-(len(name)-2)*3, y_up*30+25), font_size=9))
        self.draw_.add(self.draw_.text('θ ='+ str(np.round(theta,3)),
                                       insert=(x+55, y_up*30+30+20-6),
                                       font_size=7))
        if phi is not None:
            self.draw_.add(self.draw_.text('ϕ ='+ str(np.round(phi,3)),
                                           insert=(x+55, y_up*30+30+26-6),
                                           font_size=7))

    def draw_ps(self, order, wires, theta, name=None):
        """
        Draw phaseshif (rotation) gate.
        """
        fill_c = info_dic[name][0]
        shift= info_dic[name][1]
        x = 90 * order + 40
        y_up = wires[0]
        # y_down = wires[1]
        self.draw_.add(self.draw_.polyline(points=[(x, y_up*30+30), (x+90, y_up*30+30)],
                                           fill='none', stroke='black', stroke_width=2))
        self.draw_.add(self.draw_.rect(insert=(x+42.5, y_up*30+25), size=(6,12), rx=0, ry=0,
                                       fill=fill_c, stroke='black', stroke_width=1.5))
        self.draw_.add(self.draw_.text(name, insert=(x+40+shift, y_up*30+20), font_size=9))
        self.draw_.add(self.draw_.text('θ ='+str(np.round(theta,3)), insert=(x+55, y_up*30+20), font_size=7))

    def draw_sq(self, order, wires, r, theta, name=None):
        """
        Draw squeezing gate, displacement gate.
        """
        x = 90 * order + 40
        y_up = wires[0]
        # y_down = wires[1]
        self.draw_.add(self.draw_.polyline(points=[(x, y_up*30+30), (x+90, y_up*30+30)],
                                          fill='none', stroke='black', stroke_width=2))
        fill_c = info_dic[name][0]
        shift= info_dic[name][1]

        self.draw_.add(self.draw_.rect(insert=(x+42.5, y_up*30+25), size=(10,12), rx=0, ry=0,
                                       fill=fill_c, stroke='black', stroke_width=1.5))
        self.draw_.add(self.draw_.text(name, insert=(x+40+shift, y_up*30+20), font_size=9))
        self.draw_.add(self.draw_.text('r ='+str(np.round(r,3)), insert=(x+55, y_up*30+18), font_size=7))
        self.draw_.add(self.draw_.text('θ ='+str(np.round(theta,3)), insert=(x+55, y_up*30+24), font_size=7))

    def draw_any(self, order, wires, name):
        """
        Draw arbitrary unitary gate.
        """
        fill_c = info_dic[name][0]
        # shift= info_dic[name][1]
        x = 90 * order + 40
        y_up = wires[0]
        h = (int(len(wires)) - 1) * 30 + 20
        for k in wires:
            self.draw_.add(self.draw_.polyline(points=[(x, k*30+30),(x+20, k*30+30)],
                                               fill='none', stroke='black', stroke_width=2))

            self.draw_.add(self.draw_.polyline(points=[(x+70, k*30+30),(x+90, k*30+30)],
                                               fill='none', stroke='black', stroke_width=2))

        self.draw_.add(self.draw_.rect(insert=(x+20, y_up*30+20), size=(50, h), rx=0, ry=0,
                                       fill=fill_c, stroke='black', stroke_width=2))
        self.draw_.add(self.draw_.text(name, insert=((x+41), y_up*30+20+h/2), font_size=10))

    def draw_lines(self, order, wires):
        """
        Act nothing.
        """
        x = 90 * order + 40
        for k in wires:
            self.draw_.add(self.draw_.polyline(points=[(x, k*30+30),(x+90, k*30+30)],
                                               fill='none', stroke='black', stroke_width=2))

class DrawClements():
    """Draw the n-mode Clements architecture.

    Args:
        nmode (int): The number of modes of the Clements architecture.
        mzi_info (Dict): The dictionary for mzi parameters, resulting from the decompose function.
        cl (str, optional): The color for plotting. Default: ``'dodgerblue'``
        fs (int, optional): The fontsize. Default: 30
        method (str, optional): The way for Clements decomposition, ``'cssr'`` or ``'cssl'``.
            Default: ``'cssr'``
    """
    def __init__(
        self,
        nmode: int,
        mzi_info: Dict,
        cl: str = 'dodgerblue',
        fs: int = 30,
        method: str = 'cssr'
    ) -> None:
        self.nmode = nmode
        self.method = method
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
        """Plot  clements structure with ``cssr`` or ``cssl`` type."""
        if self.method == 'cssr':
            assert(self.nmode%2 == 0), 'plotting only valid for even modes'
            self.plotting_clements_1()
        if self.method == 'cssl':
            self.plotting_clements_2()

    def plotting_clements_1(self):
        """
        Plot ``cssr`` with left to right order.
        """
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8*3,5*3)
        # plt.rcParams['figure.figsize'] = (8*3,5.0*3)
        coords1 = []
        coords2 = []
        nmode = self.nmode
        phase_angle = self.phase_angle
        fs = self.fontsize
        cl = self.color
        wid = self.wid
        height = self.height
        for i in range(nmode):
            plt.annotate('',xy=(-0.1,1-0.25*i),
                        xytext=(-0.5,1-0.25*i),
                        arrowprops={'arrowstyle': '-|>', 'lw':5},
                        va = 'center',)
            plt.text(-0.8, 1-0.25*i, f'{i}', fontsize = fs )
            plt.plot([0, 1.2], [1-0.25*i,1-0.25*i], color = cl)
            plt.text( 3.2*(nmode/2-1)+2.2+2.1, 1-0.25*i+0.05, f'{phase_angle[i]:.3f}', fontsize=fs-8 )  # phase angle
            ax.add_patch(
            patches.Rectangle(
                (3.2*(nmode/2-1)+2.2+2.1, 1-0.25*i-0.05),
                wid,
                height,
                edgecolor = 'green',
                facecolor = 'green',
                fill=True,
                zorder=3) )  ## for PS
            if nmode%2==1:
                plt.plot([2.2+3.2*(int((nmode+1)/2)-1),
                          3.2*int((nmode+1)/2-1)+2.2+2.2],
                          [1-0.25*i,1-0.25*i],
                          color = cl)
        if nmode%2==0:   # for even mode
            for i in range(int(nmode/2)):
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1,1], color = cl)
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1-0.25*(nmode-1), 1-0.25*(nmode-1) ], color = cl)
                for j in range(nmode):
                    plt.plot([1.5+3.2*i, 3.2*i+1.9], [1-0.25*j,1-0.25*j], color = cl)
                    coords1.append( [1.5+3.2*i, 3.2*i+1.9, 1-0.25*j,1-0.25*j])
                    if 0<j<nmode-1:
                        plt.plot([3.1+3.2*i, 3.2*i+3.5], [1-0.25*j,1-0.25*j], color = cl)
                        coords2.append([3.1+3.2*i, 3.2*i+3.5, 1-0.25*j,1-0.25*j])
                        plt.plot([2.2+3.2*i, 3.2*i+2.8], [1-0.25*j,1-0.25*j], color = cl)
                        plt.plot([3.8+3.2*i, 3.2*i+4.4], [1-0.25*j,1-0.25*j], color = cl)
        if nmode%2==1:  # for odd mode
            for i in range(int((nmode+1)/2)):
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1,1], color = cl)
            #     plt.plot([1.2+3.2*i, 3.2*i+2.2+2.2], [1-0.25*(nmode-1), 1-0.25*(nmode-1) ], color = cl)
                for j in range(nmode):
                    if j< nmode-1: # remove last line
                        plt.plot([1.5+3.2*i, 3.2*i+1.9], [1-0.25*j,1-0.25*j], color = cl)
                        coords1.append( [1.5+3.2*i, 3.2*i+1.9, 1-0.25*j,1-0.25*j])
                    if j >= nmode-1:
                        plt.plot([1.2+3.2*i, 3.2*i+2.2], [1-0.25*j,1-0.25*j], color = cl)
                    if  i< int((nmode+1)/2)-1 and 0<j<nmode: # remove the last column
                        plt.plot([3.1+3.2*i, 3.2*i+3.5], [1-0.25*j,1-0.25*j], color = cl)
                        coords2.append([3.1+3.2*i, 3.2*i+3.5, 1-0.25*j,1-0.25*j])
                        plt.plot([2.2+3.2*i, 3.2*i+2.8], [1-0.25*j,1-0.25*j], color = cl)
                        plt.plot([3.8+3.2*i, 3.2*i+4.4], [1-0.25*j,1-0.25*j], color = cl)
        # connecting lines i, i+1
        for i  in range(len(coords1)):
            if i%2==0:
                self.connect1(coords1[i], ax, a=-0.5-0.4, c=0.7-0.7, cl=self.color)
            if i%2==1:
                self.connect2(coords1[i], cl=self.color)
        for i  in range(len(coords2)):
            if i%2==0:
                self.connect1(coords2[i], ax, a=-0.5-0.4, c=0.7-0.7, cl=self.color)
            if i%2==1:
                self.connect2(coords2[i], cl=self.color)
        # plotting paras
        self.plot_paras_1(self.dic_mzi, fs=self.fontsize-8)
        plt.axis(self.axis_off)
        # if self.axis_off:
        #     plt.axis('off')
        plt.show()

    def plotting_clements_2(self):
        """
        Plot ``cssl`` with right to left order.
        """
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(8*3,5*3)
        # plt.rcParams['figure.figsize'] = (8*3,5.0*3)
        coords1 = []
        coords2 = []
        nmode = self.nmode
        phase_angle = self.phase_angle
        fs = self.fontsize
        cl = self.color
        wid = self.wid
        height = self.height
        for i in range(nmode):
            plt.annotate('',xy=(-0.1,1-0.25*i),
                         xytext=(-0.5,1-0.25*i),
                         arrowprops={'arrowstyle': '-|>', 'lw':5},
                         va='center',)
            plt.text(-0.8, 1-0.25*i, f'{i}', fontsize = fs )
            plt.plot([0, 1.2], [1-0.25*i,1-0.25*i], color = cl)
            plt.text( 0.4,1-0.25*i+0.05, f'{phase_angle[i]:.3f}', fontsize=fs-8 )  # phase angle
            ax.add_patch(
            patches.Rectangle(
                (0.5,1-0.25*i-0.05),
                wid,
                height,
                edgecolor = cl,
                facecolor = cl,
                fill=True
                             ) )
            if nmode%2==1:
                plt.plot([2.2+3.2*(int((nmode+1)/2)-1),
                          3.2*int((nmode+1)/2-1)+2.2+2.2],
                          [1-0.25*i,1-0.25*i],
                          color = cl)
        if nmode%2==0:   # for even mode
            for i in range(int(nmode/2)):
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1,1], color = cl)
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1-0.25*(nmode-1), 1-0.25*(nmode-1) ], color = cl)
                for j in range(nmode):
                    plt.plot([1.5+3.2*i, 3.2*i+1.9], [1-0.25*j,1-0.25*j], color = cl)
                    coords1.append([1.5+3.2*i, 3.2*i+1.9, 1-0.25*j,1-0.25*j])
                    if 0<j<nmode-1:
                        plt.plot([3.1+3.2*i, 3.2*i+3.5], [1-0.25*j,1-0.25*j], color = cl)
                        coords2.append([3.1+3.2*i, 3.2*i+3.5, 1-0.25*j,1-0.25*j])
                        plt.plot([2.2+3.2*i, 3.2*i+2.8], [1-0.25*j,1-0.25*j], color = cl)
                        plt.plot([3.8+3.2*i, 3.2*i+4.4], [1-0.25*j,1-0.25*j], color = cl)
        if nmode%2==1:  # for odd mode
            for i in range(int((nmode+1)/2)):
                plt.plot([2.2+3.2*i, 3.2*i+2.2+2.2], [1,1], color = cl)
            #     plt.plot([1.2+3.2*i, 3.2*i+2.2+2.2], [1-0.25*(nmode-1), 1-0.25*(nmode-1) ], color = cl)
                for j in range(nmode):
                    if j< nmode-1: # remove last line
                        plt.plot([1.5+3.2*i, 3.2*i+1.9], [1-0.25*j,1-0.25*j], color = cl)
                        coords1.append( [1.5+3.2*i, 3.2*i+1.9, 1-0.25*j,1-0.25*j])
                    if j >= nmode-1:
                        plt.plot([1.2+3.2*i, 3.2*i+2.2], [1-0.25*j,1-0.25*j], color = cl)
                    if  i< int((nmode+1)/2)-1 and 0<j<nmode: # remove the last column
                        plt.plot([3.1+3.2*i, 3.2*i+3.5], [1-0.25*j,1-0.25*j], color = cl)
                        coords2.append([3.1+3.2*i, 3.2*i+3.5, 1-0.25*j,1-0.25*j])
                        plt.plot([2.2+3.2*i, 3.2*i+2.8], [1-0.25*j,1-0.25*j], color = cl)
                        plt.plot([3.8+3.2*i, 3.2*i+4.4], [1-0.25*j,1-0.25*j], color = cl)
        # connecting lines i, i+1
        for i  in range(len(coords1)):
            if i%2==0:
                self.connect1(coordinate=coords1[i], ax=ax, cl=self.color)
            if i%2==1:
                self.connect2(coordinate=coords1[i], ax=ax, cl=self.color)
        for i  in range(len(coords2)):
            if i%2==0:
                self.connect1(coordinate=coords2[i], ax=ax, cl=self.color)
            if i%2==1:
                self.connect2(coordinate=coords2[i], ax=ax, cl='black')
        # plotting paras
        self.plot_paras(self.dic_mzi, self.nmode, fs=self.fontsize-8)
        plt.axis(self.axis_off)
        # if self.axis_off:
        #     plt.axis('off')
        plt.show()

    def sort_mzi(self):
        """
        Sort mzi parameters in the same array for plotting.
        """
        dic_mzi = defaultdict( list) #当key不存在时对应的value是[]
        mzi_list = self.mzi_info['MZI_list']
        for i in mzi_list:
            dic_mzi[tuple(i[0:2])].append(i[2:])
        return dic_mzi

    def ps_pos(self):
        """
        Label the position of each phaseshifter for ``cssr`` case.
        """
        if self.method == 'cssr':
            dic_pos = { }
            nmode = self.nmode
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
    def connect1(coordinate, ax, cl, wid=0.1, height=0.08, a=-0.05, b=-0.05, c=0.7, d=-0.05):
        """
        Connect odd column.
        """
        x0, x1, y0, y1 = coordinate
    #     print(x0,x1,y0,y1)
        plt.plot([x0, x0-0.3],[y0, y0-0.25], color = cl)
        plt.plot([x1, x1+0.3],[y1, y1-0.25], color = cl)
        ax.add_patch(patches.Rectangle(
            ((x0+x1)/2 + a, y0 + b),
            wid,
            height,
            edgecolor = cl,
            facecolor = cl,
            fill=True
        ) )
        ax.add_patch(patches.Rectangle(
            ((x0+x1)/2 + c, y0 + d),
            wid,
            height,
            edgecolor = cl,
            facecolor = cl,
            fill=True
        ) )

    @staticmethod
    def connect2(coordinate, cl):
        """
        Connect even column.
        """
        x0, x1, y0, y1 = coordinate
        plt.plot([x0, x0-0.3],[y0, y0+0.25], color = cl)
        plt.plot([x1, x1+0.3],[y1, y1+0.25], color = cl)

    @staticmethod
    def plot_paras(sort_mzi_dic, nmode, fs=20):
        """
        Plot mzi parameters for ``cssl`` case.
        """
        for i in sort_mzi_dic.keys():
            if i[0]%2 == 0: # 0, 2, 4, 6..
                temp_values = sort_mzi_dic[i]
                len_ = len(temp_values)
                for j in range(len_):
                    plt.text(8.6-3.2*j+3.2*((nmode-6)//2+nmode%2),
                             1-0.25*i[0]+0.05,
                             f'{temp_values[j][0]:.3f}',
                             fontsize=fs)
                    plt.text(7.8-3.2*j+3.2*((nmode-6)//2+nmode%2),
                             1-0.25*i[0]+0.05,
                             f'{temp_values[j][1]:.3f}',
                             fontsize=fs)
            if i[0]%2 ==1: # 1, 3..
                temp_values = sort_mzi_dic[i]
                len_ = len(temp_values)
                for j in range(len_):
                    plt.text(8.6-3.2*j+1.6+3.2*((nmode-6)//2),
                             1-0.25*i[0]+0.05,
                             f'{temp_values[j][0]:.3f}',
                             fontsize=fs)
                    plt.text(7.8-3.2*j+1.6+3.2*((nmode-6)//2),
                             1-0.25*i[0]+0.05,
                             f'{temp_values[j][1]:.3f}',
                             fontsize=fs)

    @staticmethod
    def plot_paras_1(sort_mzi_dic, fs=20):
        """
        Plot mzi parameters for ``cssr`` case.
        """
        for i in sort_mzi_dic.keys():
            if i[0]%2 == 0: # 0, 2, 4, 6..
                temp_values = sort_mzi_dic[i]
                len_ = len(temp_values)
                for j in range(len_):
                    plt.text(3.2*j+0.6, 1-0.25*i[0]+0.05, f'{temp_values[j][0]:.3f}', fontsize=fs)
                    plt.text(3.2*j+0.6+0.9, 1-0.25*i[0]+0.05, f'{temp_values[j][1]:.3f}', fontsize=fs)
            if i[0]%2 ==1: # 1, 3..
                temp_values = sort_mzi_dic[i]
                len_ = len(temp_values)
                for j in range(len_):
                    plt.text(3.2*j+0.6+1.6, 1-0.25*i[0]+0.05, f'{temp_values[j][0]:.3f}', fontsize=fs)
                    plt.text(3.2*j+0.6+2.4, 1-0.25*i[0]+0.05, f'{temp_values[j][1]:.3f}', fontsize=fs)
