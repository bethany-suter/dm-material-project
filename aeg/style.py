from matplotlib import rc, rcParams
import seaborn as sns


rc('text', usetex=True)
rc('text.latex',
    preamble=(
        r'\usepackage{amsmath}\usepackage{siunitx}'
        r'\DeclareSIUnit{\year}{yr}'
        r'\usepackage{bm}'
        r'\newcommand{\bb}[1]{\bm{\mathrm{#1}}}'
    )
)
# Change all fonts to 'Computer Modern'
rc('font', **{'size':14, 'family':'serif','serif':['Times New Roman']})
rc('xtick.major', size=5, pad=7)
rc('xtick', labelsize=18)
rc('ytick.major', size=5, pad=7)
rc('ytick', labelsize=18)
rcParams['figure.figsize'] = 8, 6
rcParams['axes.labelsize'] = 20
rcParams['axes.titlesize'] = 20
rcParams['legend.loc'] = 'best'
rcParams['legend.fontsize'] = 16
rcParams['legend.frameon'] = False


FIG_SINGLE_SIZE = (8,  6)
FIG_DOUBLE_SIZE = (16, 6)

COLORS = sns.color_palette('colorblind')
