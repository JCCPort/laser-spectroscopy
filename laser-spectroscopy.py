import datetime
import math
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uncertainties
from lmfit.models import PseudoVoigtModel, LinearModel
from scipy.optimize import curve_fit

matplotlib.use('Agg')
today = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
print(today)

os.chdir('C:\\Users\Josh\Desktop\LSPEC1\ReadableData')


class RangeTool(object):
    """
    Like Cursor but the crosshair snaps to the nearest x,y point
    For simplicity, I'm assuming x is sorted
    """

    def __init__(self, ax, data, key, figure2):
        self.ax = figure2.axes
        self.data = data
        self.key = key
        self.figure2 = figure2
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.lowers = np.array([])
        self.uppers = np.array([])
        self.IndependentVariable = "Frequency (a.u)"
        self.DependentVariable = "Intensity (a.u)"
        self.x = data[self.IndependentVariable]
        self.y = data[self.DependentVariable]
        self.ax.set_xlim(np.min(self.x), np.max(self.x))
        height = np.max(self.y) - np.min(self.y)
        self.ax.set_ylim(np.min(self.y)-0.1*height, np.max(self.y)+0.1*height)
        # text location in axes coords
        self.txt = ax.text(0.7, 0.9, '', transform=ax.transAxes)
        self.cid1 = figure2.figure.canvas.mpl_connect('key_press_event', self.rangeselect)
        self.cid2 = figure2.figure.canvas.mpl_connect('motion_notify_event', self.mouse_move)
        self.cid3 = figure2.figure.canvas.mpl_connect('key_press_event', self.rangeremove)
        self.cid4 = figure2.figure.canvas.mpl_connect('key_press_event', self.finishplot)
        self.Ranges = pd.DataFrame(columns=['Lower Bound', 'LowerIndex', 'Upper Bound', 'UpperIndex', 'Displayed'])
        self.il = 0
        self.iu = 0
        self.t = 0

    def __call__(self, event):
        print('click', event)
        print(event.xdata, event.ydata)
        if event.inaxes != self.figure2.axes: return

    def mouse_move(self, event):

        if not event.inaxes:
            return

        x, y = event.xdata, event.ydata
        indx = min(np.searchsorted(self.x, [x])[0], len(self.x) - 1)
        x = self.x[indx]
        y = self.y[indx]
        # update the line positions
        self.lx.set_ydata(y)
        self.ly.set_xdata(x)
        # print('{},{}'.format(event.xdata, event.ydata))
        self.txt.set_text('x=%1.2f, y=%1.2f' % (x, y))
        self.figure2.figure.canvas.draw_idle()
        # print('x=%1.2f, y=%1.2f' % (x, y))

    def rangeselect(self, event):
        if not event.inaxes:
            return
        x = event.xdata
        print(x)
        indx = min(np.searchsorted(self.x, [x])[0], len(self.x) - 1)
        x = self.x[indx]
        if event.key == 'tab':
            self.Ranges.at[self.il, 'Lower Bound'] = x
            self.Ranges.at[self.il, 'LowerIndex'] = indx
            self.il += 1
        if event.key == 'shift':
            self.Ranges.at[self.iu, 'Upper Bound'] = x
            self.Ranges.at[self.iu, 'UpperIndex'] = indx
            self.iu += 1
        if self.il == self.iu:
            try:
                if math.isnan(self.Ranges.at[self.il-1, 'Displayed']):
                    self.ax.axvspan(self.Ranges.at[self.il - 1, 'Lower Bound'],
                                    self.Ranges.at[self.iu - 1, 'Upper Bound'],
                                    alpha=0.1, edgecolor='k', linestyle='--')
                    self.Ranges.at[self.il-1, 'Displayed'] = 1
            except ValueError:
                pass
        self.cid3 = self.figure2.figure.canvas.mpl_connect('key_press_event', self.rangeremove)

    def rangeremove(self, event):
        if not event.inaxes:
            return
        if event.key == 'delete':
            if not self.Ranges.empty:
                self.figure2.figure.canvas.mpl_disconnect(self.cid1)
                try:
                    self.Ranges.at[self.il - 1, 'Displayed'] = float('NaN')
                    self.il -= 1
                    self.iu -= 1
                    self.Ranges.drop(self.Ranges.index[-1], inplace=True)
                    Polys = self.ax.get_children()
                    Polys[len(self.Ranges.index)].remove()
                except IndexError:
                    self.Ranges.at[self.il - 1, 'Displayed'] = float('NaN')
                    self.il -= 1
                    self.iu -= 1
                    self.Ranges.drop(self.Ranges.index[0], inplace=True)
                    Polys = self.ax.get_children()
                    Polys[0].remove()
                    if self.Ranges == 'Empty DataFrame':
                        print('Range list is empty')
                # except NotImplementedError:
                #     Polys[len(self.Ranges.index)] = Polys(alpha=0)[len(self.Ranges.index)]
                finally:
                    pass
                self.cid1 = self.figure2.figure.canvas.mpl_connect('key_press_event', self.rangeselect)

    def finishplot(self, event):
        if event.key == 'enter':
            os.chdir('C:\\Users\Josh\Desktop\LSPEC1\Ranges')
            self.Ranges.to_csv('{}.csv'.format(self.key), index=False, encoding='utf-8',
                               columns=['Lower Bound', 'LowerIndex', 'Upper Bound', 'UpperIndex'])
            plt.close()
            os.chdir('C:\\Users\Josh\Desktop\LSPEC1\ReadableData')
        if event.key == 'escape':
            plt.close()
        # print('\n')
        # print('Ranges are \n {}'.format(self.Ranges))


Data = {}


class DataRead:
    def __init__(self, exp, peak, run, end):
        self.IndependentVariable = "Frequency (a.u)"
        self.DependentVariable = "Intensity (a.u)"
        self.peak = peak
        self.run = run
        self.exp = exp
        self.end = end
        self.files = []
        self.xrange = []
        self.yrange = []
        self.xranges = {}
        self.yranges = {}
        self.datpath = 'ReadableData'
        self.rangepath = 'Ranges'
        self.rangename = pd.read_csv('{}.csv'.format(self.datafilename))
        os.chdir('C:\\Users\Josh\Desktop\LSPEC1\{}'.format(self.datpath))
        self.datafilename = '{}_{}_{}_{}'.format(self.exp, self.peak, self.run, self.end)
        self.dataset = self.datafilename.split('.')[0]
        self.dataset = pd.read_csv('{}.csv'.format(self.datafilename),
                                   header=None, delimiter=',',
                                   names=[self.IndependentVariable, self.DependentVariable],
                                   float_precision='round_trip', engine='c')
        print(self.dataset)
        self.dataset['Frequency (a.u)'] = self.dataset['Frequency (a.u)'].apply(lambda x: (x*((380/0.002610666056666669))))
        print(self.dataset)

    def range(self):
        for i in range(0, len(self.rangename)):
            self.xrange.append((self.dataset[self.IndependentVariable][self.rangename['LowerIndex'][i]:self.rangename['UpperIndex'][i]+1]).values)
            self.yrange.append((self.dataset[self.DependentVariable][self.rangename['LowerIndex'][i]:self.rangename['UpperIndex'][i]+1]).values)
        for i in range(0, len(self.xrange)):
            self.xranges[i] = self.xrange[i]
            self.yranges[i] = self.yrange[i]
        return self.xranges, self.yranges, self.xrange, self.yrange

    def singleplot(self):
        start_time = time.time()

        os.chdir('C:\\Users\Josh\Desktop\LSPEC1\{}'.format(self.rangepath))
        self.range()
        A = []
        B = []

        a_87 = {0: "$F=2 \Rightarrow F'=3$",
                1: "Unknown trough",
                2: "Crossover: $F'=2, F'=3$",
                3: "Crossover: $F'=1, F'=3$",
                4: "$F=2 \Rightarrow F'=2$",
                5: "Crossover: $F'=2, F'=1$",
                6: "$F=2 \Rightarrow F'=1$"}
        a_85 = {0: "$F=3 \Rightarrow F'=4$",
                1: "Crossover: $F'=3, F'=4$",
                2: "Crossover: $F'=2, F'=4$",
                3: "$F=3 \Rightarrow F'=3$",
                4: "Crossover: $F'=3, F'=2$",
                5: "$F=3 \Rightarrow F'=2$"}
        b_87 = {0: "$F=1 \Rightarrow F'=2$",
                1: "Crossover: $F'=1, F'=2$",
                2: "Crossover: $F'=0, F'=2$",
                3: "$F=1 \Rightarrow F'=1$",
                4: "Crossover: $F'=1, F'=0$",
                5: "$F=1 \Rightarrow F'=0$"}
        b_85 = {0: "$F=2 \Rightarrow F'=3$",
                1: "Crossover: $F'=2, F'=3$",
                2: "Crossover: $F'=1, F'=3$",
                3: "$F=2 \Rightarrow F'=2$",
                4: "Crossover: $F'=1, F'=2$",
                5: "$F=2 \Rightarrow F'=1$"}

        dictdict = {1: b_87, 2: b_85, 3: a_85, 4: a_87}
        self.fitvals = pd.DataFrame(columns=['Peak', 'Hyperfine transition',
                                             'Relative frequency (MHz)', 'FWHM (MHz)',
                                             'Fraction'])
        for i in range(0, len(self.xrange)):
            A.append(self.xranges[i][np.where(self.yranges[i] == np.min(self.yranges[i]))[0]][0])
            B.append(A[i]-A[i-1])
            # C = np.mean(B[1:])
            # print('A', A, 'A')
            # print('B', B, 'B')
            # print('C', C, 'C')
            # print(self.xranges[i])
            voigt_mod = PseudoVoigtModel(prefic='voigt_')
            line_mod = LinearModel(prefix='line_')
            params = line_mod.make_params(intercept=np.min(self.yrange[i]))
            params += voigt_mod.guess(self.yranges[i], x=self.xranges[i], center=np.median(self.xranges[i]),
                                      sigma=np.std(self.xranges[i]), height=np.max(self.yranges[i]), fraction=0.7)
            model = voigt_mod + line_mod
            # params['gamma'].set(value=np.std(self.xranges[i]), vary=True)
            result = model.fit(self.yranges[i], params, x=self.xranges[i])
            labels = dictdict[self.peak]
            plt.plot(self.xranges[i], result.best_fit, antialiased=True, label='\n' + labels[i] + '\n${0:.2f} MHz$'.format(result.params['center'].value))
            # dely = result.eval_uncertainty(sigma=1)
            # plt.fill_between(self.xranges[i], result.best_fit - dely, result.best_fit + dely, color="#ABABAB")
            plt.legend(fancybox=True, mode='expand', handlelength=0.01)
            centers = uncertainties.ufloat(result.params['center'].value, result.params['center'].stderr)
            fwhms = uncertainties.ufloat(result.params['fwhm'].value, result.params['fwhm'].stderr)
            fractions = uncertainties.ufloat(result.params['fraction'].value, result.params['fraction'].stderr)
            peakdict = {1: '87b', 2: '85b', 3: '85a', 4: '87a'}
            self.fitvals = self.fitvals.append({
                                      'Hyperfine transition': labels[i],
                                      'Relative frequency (MHz)': centers,
                                      'FWHM (MHz)': fwhms,
                                      'Fraction': fractions}, ignore_index=True)
            # self.texwriter()
            print(labels[i])
            print(result.fit_report())
            print('----{}----'.format(time.time() - start_time))
        # fitvals.to_csv('fit_{}_{}.csv'.format(self.peak, self.run))

    def texwriter(self):
        os.chdir('C:\\Users\Josh\Desktop\LSPEC1\LATEX')
        with open('{}_{}_{}_{}.tex'.format(self.exp, self.peak, self.run, self.end), 'w') as texfile:
            texfile.write('\\documentclass[a4paper,twoside]{article}\n')
            texfile.write('\\usepackage[margin=0in]{geometry}\n')
            texfile.write('\\usepackage{mathtools}\n')
            texfile.write('\\usepackage[math]{cellspace}\n')
            texfile.write('\\pagenumbering{gobble}\n')
            texfile.write('\\cellspacetoplimit 4pt\n')
            texfile.write('\\cellspacebottomlimit 4pt\n')
            texfile.write('\n')
            texfile.write(r'\setlength{\topmargin}{1in}')
            texfile.write('\n')

            texfile.write('\\begin{document}\n')

            # texfile.write(r' \newline')
            # texfile.write('\\ \n')
            texfile.write(r'''\ {\def\arraystretch{1.2}\tabcolsep=10pt''')
            texfile.write('\\ \n')
            texfile.write('\\begin{tabular}[h!]{|l|l|l|l|} \n')
            texfile.write('\hline')
            row_fields = ('Hyperfine transition', 'Frequency (MHz)', 'FWHM (MHz)',
                          'Fraction')
            texfile.write('\\ {} & {} & {} & {} \\\\ \n'.format(row_fields[0], row_fields[1],
                                                                row_fields[2], row_fields[3]))
            texfile.write('\hline \hline')
            for i in range(0, len(self.fitvals['Peak'])):
                texfile.write('\\ {} & ${:2L}$ & ${:5L}$ & ${:1L}$ \\\\ \n'.format(
                                                                                      self.fitvals['Hyperfine transition'][i],
                                                                                      self.fitvals['Relative frequency (MHz)'][i],
                                                                                      self.fitvals['FWHM (MHz)'][i],
                                                                                      self.fitvals['Fraction'][i]))
            texfile.write('\hline')
            texfile.write('\\end{tabular}\n')
            texfile.write('\\ }\n')
            texfile.write('\\end{document}\n')
            os.chdir('C:\\Users\Josh\Desktop\LSPEC1')

    def multi_plot(self):
        os.chdir('C:\\Users\Josh\Desktop\LSPEC1\{}'.format(self.rangepath))
        self.rangename = pd.read_csv('{}.csv'.format(self.datafilename), engine='c')
        self.range()
        for i in range(0, len(self.xrange)):
            voigt_mod = PseudoVoigtModel(prefic='voigt_')
            line_mod = LinearModel(prefix='line_')
            params = line_mod.make_params(intercept=np.min(self.yrange[i]))
            params += voigt_mod.guess(self.yranges[i], x=self.xranges[i], center=np.median(self.xranges[i]),
                                      sigma=np.std(self.xranges[i]), height=np.max(self.yranges[i]), fraction=0.7)
            model = voigt_mod + line_mod
            result = model.fit(self.yranges[i], params, x=self.xranges[i])

            fig = plt.figure()
            fig.subplots_adjust(hspace=0.3, wspace=0)
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(self.xranges[i], result.best_fit, antialiased=True)
            ax1.plot(self.xranges[i], self.yranges[i], '.', color='#1c1c1c')
            dely = result.eval_uncertainty(sigma=1)
            ax1.fill_between(self.xranges[i], result.best_fit - dely, result.best_fit + dely, color="#ABABAB")
            ax1.grid(color='k', linestyle='--', alpha=0.2)
            plt.title('Peak with 1 sigma error bands')

            ax2 = fig.add_subplot(2, 2, 2)
            ax2.plot(self.xranges[i], self.yranges[i]-result.best_fit, '.', antialiased=True)
            ax2.grid(color='k', linestyle='--', alpha=0.2)
            plt.title('Residuals')

            ax3 = fig.add_subplot(2, 2, 3)
            ax3.plot(self.xranges[i], ((self.yranges[i]-result.best_fit)**2)/(dely**2), '.', antialiased=True)
            ax3.grid(color='k', linestyle='--', alpha=0.2)
            plt.title('Normalised residuals')

            ax4 = fig.add_subplot(2, 2, 4)
            ax4.hist(self.yranges[i]-result.best_fit, bins=15)
            ax4.grid(color='k', linestyle='--', alpha=0.2)
            plt.title('Residual histogram')

            fig.tight_layout()
            fig.set_size_inches(16.5, 10.5)
            fig_manager = plt.get_current_fig_manager()
            fig_manager.window.showMaximized()
            fig.suptitle(('Peak {0:.0f}'.format(self.peak) + ' ' +
                          'Run {0:.0f}'.format(self.run) + ' ' +
                          'Hyperfine peak at {0:.5f}'.format(result.params['center'].value)))
            print(self.peak, self.run, result.params['center'].value)
            print('Peak {0:2d}, Run {0:.0f}, Hyperfine peak at {0:.5f}'.format(self.peak, self.run, result.params['center'].value))
            plt.show()
            print(result.fit_report())


def DataConvert(datafolder, destinationfolder):
    os.chdir('C:\\Users\Josh\Desktop\LSPEC1\{}'.format(datafolder))
    for filename in os.listdir(os.getcwd()):
        print(filename)
        name = filename.split('.')[0]
        nam2 = name.split('_')
        nam3 = nam2[0] + '_' + nam2[1] + '_' + nam2[2] + '_' + ('R'+nam2[3])
        if filename.split('.')[1] == 'csv':
            try:
                dframename = pd.read_csv(filename, header=None, delimiter=',', usecols=[9, 10], engine='c')
            except ValueError:
                dframename = pd.read_csv(filename, header=None, delimiter=',', usecols=[3, 4], engine='c')

            os.chdir('C:\\Users\Josh\Desktop\LSPEC1\{}'.format(destinationfolder))
            dframename.to_csv('{}.csv'.format(nam3), header=False, index=False)
            os.chdir('C:\\Users\Josh\Desktop\LSPEC1\{}'.format(datafolder))
        elif filename.split('.')[1] == 'xlsx':
            try:
                dframename2 = pd.read_excel(filename, usecols='J:K', engine='c')
            except ValueError:
                dframename2 = pd.read_excel(filename, usecols='D:E', engine='c')
            print(dframename2)
            os.chdir('C:\\Users\Josh\Desktop\LSPEC1\{}'.format(destinationfolder))
            dframename2.to_csv('{}.csv'.format(nam3), header=False, index=False)
            os.chdir('C:\\Users\Josh\Desktop\LSPEC1\{}'.format(datafolder))


def plot_outputter(exp, peak, run, end):
    peakdict = {1: '87b', 2: '85b', 3: '85a', 4: '87a'}
    fig, ax = plt.subplots()
    figure2, = ax.plot(DataRead(exp, peak, run, end).dataset[DataRead(exp, peak, run, end).IndependentVariable],
                       DataRead(exp, peak, run, end).dataset[DataRead(exp, peak, run, end).DependentVariable],
                       '.', antialiased='True', color='#1c1c1c', mew=1.0, markersize=2.5)
    # thing = RangeTool(ax, DataRead(exp, peak, run, end).dataset, DataRead(exp, peak, run, end).datafilename, figure2)
    plt.ylabel('Intensity (a.u)')
    plt.xlabel('Frequency (a.u)')
    plt.title('{} {}'.format(peakdict[peak], end))
    ax.grid(color='k', linestyle='--', alpha=0.2)
    fig.set_size_inches(13.5, 10.5)

    print('\n')
    if os.path.isfile('C:\\Users\Josh\Desktop\LSPEC1\Ranges\{}_{}_{}_{}.csv'.format(exp, peak, run, end)):
        DataRead(exp, peak, run, end).singleplot()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    os.chdir('C:\\Users\Josh\Desktop\LSPEC1\Figures')
    # plt.savefig('{}_{}_{}_{}.png'.format(exp, peak, run, end), dpi=600)
    os.chdir('C:\\Users\Josh\Desktop\LSPEC1')

    plt.show()


def multi_plot_outputter():
    fig, axes = plt.subplots(nrows=2, ncols=2)


def calib():
    os.chdir('C:\\Users\Josh\Desktop\LSPEC1')

    separations = ['87a_2-1', '87a_3-2', '87a_3-1', '87b_1-0', '87b_2-1', '87b_2-0',
                   '85a_3-2', '85a_4-3', '85a_4-2', '85b_2-1', '85b_3-2', '85a_3-1']
    sepvals = [156.947, 266.650, 156.947+266.650, 72.218, 156.947, 156.947+72.218,
               63.401, 120.640, 120.640+63.401, 29.372, 63.401, 63.401+29.372]
    sepuncert = [0.0007, 0.0009, 0.0007+0.0009, 0.0004, 0.0007, 0.0007+0.0004,
                 0.00061, 0.00068, 0.00061+0.00068, 0.00090, 0.00061, 0.00090+0.00061]
    sepframe = pd.DataFrame({'Transitions': separations,
                             'Separations (MHz)': sepvals,
                             'Uncertainties (MHz)': sepuncert
                             })

    expsepvals = [159.1307098, 275.1126678, 434.2433776, 78.98260971, 170.6878434, 249.6704531,
                  62.96421957, 116.513182, 179.4774018, 28.38235498, 60.14233879, 88.52469377]
    expsepuncert = [0.071507349, 0.277357229, 0.32113452, 0.190950673, 0.048967155, 0.199222102,
                    0.068012705, 0.164305715, 0.211494546, 0.247178121, 0.019701421, 0.240733956]
    expsepframe = pd.DataFrame({'Transitions': separations,
                             'Separations (MHz)': expsepvals,
                             'Uncertainties (MHz)': expsepuncert
                             })

    x, y = sepframe['Separations (MHz)'], expsepframe['Separations (MHz)']
    xerr, yerr = expsepframe['Uncertainties (MHz)'], sepframe['Uncertainties (MHz)']
    el_x = np.zeros(shape=(len(x), 100))
    el_y = np.zeros(shape=(len(y), 100))
    for k in range(0, len(x)):
        xs = []
        ys = []
        for t in np.linspace(0, 2*np.pi, 100):
            xs.append(x[k] + xerr[k] * np.cos(t))
            ys.append(y[k] + yerr[k] * np.sin(t))
        # print(xs)
        el_x[k] = xs
        el_y[k] = ys

    def g2(v, m, c):
        return m * v + c
    popt, pcov = curve_fit(g2, x, ydata=y,
                           sigma=yerr, absolute_sigma=True)
    popt2, pcov2 = curve_fit(g2, x, y-g2(x, *popt))

    print(popt, pcov)
    print(popt2)
    plt.plot(np.linspace(0, 450, 500), g2(np.linspace(0, 450, 500), *popt))
    plt.plot(x, y, 'x')
    for i in range(0, len(el_x)):
        plt.plot(el_x[i], el_y[i], '.', color='k', mew=0.5, markersize=0.3, antialiased=True)
    plt.grid(color='k', linestyle='--', alpha=0.2)
    plt.ylabel('Experimental Separations (MHz)')
    plt.xlabel('Accepted Separations (MHz)')
    fig = plt.gcf()
    fig.set_size_inches(13.5, 10.5)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    # plt.savefig('LSPEC1_CALIB_1.png', dpi=600)
    plt.show()

    plt.plot(x, y-g2(x, *popt), 'x')
    plt.plot(x, g2(x, *popt2))
    plt.grid(color='k', linestyle='--', alpha=0.2)
    plt.xlabel('Experimental Separations (MHz)')
    plt.ylabel('Residuals (MHz)')
    fig = plt.gcf()
    fig.set_size_inches(13.5, 10.5)
    fig_manager = plt.get_current_fig_manager()
    fig_manager.window.showMaximized()
    # plt.savefig('LSPEC1_CALIB_RESID_1.png', dpi=600)
    plt.show()

    plt.hist(y-g2(x, *popt),  bins='auto')
    plt.show()
    os.chdir('C:\\Users\Josh\Desktop\LSPEC1\ReadableData')


for j in {'R0', 'R5', 'R10', 'R15', 'R25', 'R35', 'R45'}:
    for i in range(1, 5):
        plot_outputter('SAT', i, 6, j)


print('Complete')
