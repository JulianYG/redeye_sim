#What it will eventually look like:
#2 plots total: Accuracy & Energy vs Gaussian Sigma, Accuracy & Energy vs Uniform Standard Deviation
#x axis represents increasing Gaussian Sigma or increasing Uniform Standard Deviation

reset
set terminal postscript eps color enhanced "Helvetica" 20

color1 = "#B7C951"
color2 = "#6351C9"
color3 = "#FF974D"
color4 = "#d34f4f"
color5 = "#ffee60"

#set style data histograms 
set style fill solid 1.0 border lt -1
set datafile separator ","
set tics out nomirror
#set xtics ("" 0, "Ladada" 10, "Hahaha" 20, "Bahahah" 30, "" 40, "" 50, "" 60, "" 70, "" 80)


set style fill transparent solid 0.8 border lc rgb '#3D3D3D'
set border 31 front linewidth 2



#set ylabel "Energy (mJ/J)”
set xrange[0:49]
set yrange [0:1]
set ylabel "AlexNet Accuracy (%)"
set ytics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set y2label "Energy (J)"
set ytics nomirror
set y2tics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set xlabel "Uniform Noise Standard Deviation"
set output "~/caffe_exp/Paper_Graphs/Output/AccuracyEnergyvsUniform_Alex.eps"
plot "~/caffe_exp/Paper_Graphs/Data/alex_qvary_1P2P3P4P5P.csv" using COL=1 with linespoints  title "Top 5 Accuracy, 1 Phase" lc rgb color1 lw 2, \
                        ''  using COL=2 with linespoints	title "Energy, 1 Phases" lc rgb color1 lw 2, \
                        ''  using COL=3 with linespoints	title "Top 5 Accuracy, 2 Phases" lc rgb color2 lw 2, \
                        ''  using COL=4 with linespoints	title "Energy, 2 Phases" lc rgb color2 lw 2, \
                        ''  using COL=5 with linespoints	title "Top 5 Accuracy, 3 Phases" lc rgb color3 lw 3, \
                        ''  using COL=6 with linespoints	title "Energy, 3 Phases" lc rgb color3 lw 3, \
                        ''  using COL=7 with linespoints	title "Top 5 Accuracy, 4 Phases" lc rgb color4 lw 4, \
                        ''  using COL=8 with linespoints	title "Energy, 4 Phases" lc rgb color4 lw 4, \
                        ''  using COL=9 with linespoints	title "Top 5 Accuracy, 5 Phases" lc rgb color5 lw 5, \
                        ''  using COL=10 with linespoints	title "Energy, 5 Phases" lc rgb color5 lw 5, \

set xlabel "Gaussian Noise Standard Deviation ({/Symbol s} {/Symbol \264} 10^-^4)"
#set xlabel "Gaussian Noise Standard Deviation ({/Symbol s})"
set xrange[0:49]
set yrange [0:1]
set ylabel "AlexNet Accuracy (%)"
set ytics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set y2label "Energy (J)"
set ytics nomirror
set y2tics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)

set output "~/caffe_exp/Paper_Graphs/Output/AccuracyEnergyvsGaussian_Alex.eps"
plot "~/caffe_exp/Paper_Graphs/Data/alex_gvary_1P2P3P4P5P.csv" using COL=1 with linespoints  title "Top 5 Accuracy, 2 Phases" lc rgb color1 lw 2, \
                        ''  using COL=2 with linespoints	title "Energy, 2 Phases" lc rgb color1 lw 2, \
                        ''  using COL=3 with linespoints	title "Top 5 Accuracy, 2 Phases" lc rgb color2 lw 2, \
                        ''  using COL=4 with linespoints	title "Energy, 2 Phases" lc rgb color2 lw 2, \
                        ''  using COL=5 with linespoints	title "Top 5 Accuracy, 3 Phases" lc rgb color3 lw 3, \
                        ''  using COL=6 with linespoints	title "Energy, 3 Phases" lc rgb color3 lw 3, \
                        ''  using COL=7 with linespoints	title "Top 5 Accuracy, 4 Phases" lc rgb color4 lw 4, \
                        ''  using COL=8 with linespoints	title "Energy, 4 Phases" lc rgb color4 lw 4, \
                        ''  using COL=9 with linespoints	title "Top 5 Accuracy, 5 Phases" lc rgb color5 lw 5, \
                        ''  using COL=10 with linespoints	title "Energy, 5 Phases" lc rgb color5 lw 5, \

#set ylabel "Energy (mJ/J)”
set xrange[0:49]
set yrange [0:1]
set ylabel "GoogLeNet Accuracy (%)"
set ytics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set y2label "Energy (J)"
set ytics nomirror
set y2tics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set xlabel "Uniform Noise Standard Deviation "
set output "~/caffe_exp/Paper_Graphs/Output/AccuracyEnergyvsUniform_Alex.eps"
plot "~/caffe_exp/Paper_Graphs/Data/alex_qvary_1P2P3P4P5P.csv" using COL=1 with linespoints  title "Top 5 Accuracy, 1 Phase" lc rgb color1 lw 2, \
                        ''  using COL=2 with linespoints        title "Energy, 1 Phases" lc rgb color1 lw 2, \
                        ''  using COL=3 with linespoints        title "Top 5 Accuracy, 2 Phases" lc rgb color2 lw 2, \
                        ''  using COL=4 with linespoints        title "Energy, 2 Phases" lc rgb color2 lw 2, \
                        ''  using COL=5 with linespoints        title "Top 5 Accuracy, 3 Phases" lc rgb color3 lw 3, \
                        ''  using COL=6 with linespoints        title "Energy, 3 Phases" lc rgb color3 lw 3, \
                        ''  using COL=7 with linespoints        title "Top 5 Accuracy, 4 Phases" lc rgb color4 lw 4, \
                        ''  using COL=8 with linespoints        title "Energy, 4 Phases" lc rgb color4 lw 4, \
                        ''  using COL=9 with linespoints        title "Top 5 Accuracy, 5 Phases" lc rgb color5 lw 5, \
                        ''  using COL=10 with linespoints       title "Energy, 5 Phases" lc rgb color5 lw 5, \

set xlabel "Gaussian Noise Standard Deviation ({/Symbol s} {/Symbol \264} 10^-^4)"
#set xlabel "Gaussian Noise Standard Deviation ({/Symbol s})"
set xrange[0:49]
set yrange [0:1]
set ylabel "GoogLeNet Accuracy (%)"
set ytics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set y2label "Energy (J)"
set ytics nomirror
set y2tics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)

set output "~/caffe_exp/Paper_Graphs/Output/AccuracyEnergyvsGaussian_Alex.eps"
plot "~/caffe_exp/Paper_Graphs/Data/alex_gvary_1P2P3P4P5P.csv" using COL=1 with linespoints  title "Top 5 Accuracy, 2 Phases" lc rgb color1 lw 2, \
                        ''  using COL=2 with linespoints        title "Energy, 2 Phases" lc rgb color1 lw 2, \
                        ''  using COL=3 with linespoints        title "Top 5 Accuracy, 2 Phases" lc rgb color2 lw 2, \
                        ''  using COL=4 with linespoints        title "Energy, 2 Phases" lc rgb color2 lw 2, \
                        ''  using COL=5 with linespoints        title "Top 5 Accuracy, 3 Phases" lc rgb color3 lw 3, \
                        ''  using COL=6 with linespoints        title "Energy, 3 Phases" lc rgb color3 lw 3, \
                        ''  using COL=7 with linespoints        title "Top 5 Accuracy, 4 Phases" lc rgb color4 lw 4, \
                        ''  using COL=8 with linespoints        title "Energy, 4 Phases" lc rgb color4 lw 4, \
                        ''  using COL=9 with linespoints        title "Top 5 Accuracy, 5 Phases" lc rgb color5 lw 5, \
                        ''  using COL=10 with linespoints       title "Energy, 5 Phases" lc rgb color5 lw 5, \



