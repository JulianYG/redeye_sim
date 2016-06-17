#What it will eventually look like:
#2 plots total: Accuracy & Energy vs Gaussian Sigma, Accuracy & Energy vs Uniform Standard Deviation
#x axis represents increasing Gaussian Sigma or increasing Uniform Standard Deviation

reset
set terminal postscript eps color enhanced "Helvetica" 20

set key above left

color6 = "#808080"

color1 = "#9f99a3" #corresponds to colors 2-6 in graphdocket
color2 = "#a88b89"
color3 = "#a1b6a4"
color4 = "#a0aeb4"
color5 = "#c4bcb2"

#set style data histograms 
set style fill solid 1.0 border lt -1
set datafile separator ","
set tics out nomirror


set style fill transparent solid 0.8 border lc rgb '#3D3D3D'
set border 31 front linewidth 2

set style line 1 dt 1 lc rgb color1 lw 5 pt 8 pointinterval 5 
set style line 2 dt 3 lc rgb color1 lw 8 pt 8 pointinterval 5
set style line 3 dt 1 lc rgb color2 lw 4 pt 2 pointinterval 5
set style line 4 dt 3 lc rgb color2 lw 7 pt 2 pointinterval 5
set style line 5 dt 1 lc rgb color3 lw 3 pt 6 pointinterval 5
set style line 6 dt 3 lc rgb color3 lw 6 pt 6 pointinterval 5
set style line 7 dt 1 lc rgb color4 lw 2 pt 4 pointinterval 5
set style line 8 dt 3 lc rgb color4 lw 5 pt 4 pointinterval 5
set style line 9 dt 1 lc rgb color5 lw 2 pt 5 pointinterval 5
set style line 10 dt 3 lc rgb color5 lw 4 pt 5 pointinterval 5
set style line 11 dt 3 lc rgb color6 lw 4 pt 0 pointinterval 5
set style line 12 dt 1 lc rgb color6 lw 4 pt 0 pointinterval 5


# set xrange[0:49]
# set yrange [0:1]
# set xlabel "Gaussian Noise SNR (dB)"
# set ylabel "Alexnet Accuracy (%)"
# set y2tics nomirror
# set autoscale y2
# set xtics ("10" 0, "15" 5, "20" 10, "25" 15, "30" 20, "35" 25, "40" 30, "45" 35, "50" 40, "55" 45, "60" 50)
# set ytics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
# set y2label "Energy (J)"
# set ytics nomirror
# #set y2tics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
# set output "~/caffe_exp/Paper_Graphs/Output/alex_qvary_SNR.eps"
# plot "~/caffe_exp/Paper_Graphs/Data/for_gnu_alex_qvary_SNR.csv" using COL=2 with linespoints  notitle ls 1 axes x1y1, \
#                         ''  using COL=3 with linespoints	notitle ls 2 axes x1y2, \
#                         ''  using COL=4 with linespoints	notitle ls 3 axes x1y1, \
#                         ''  using COL=5 with linespoints	notitle ls 4 axes x1y2, \
#                         ''  using COL=6 with linespoints	notitle ls 5 axes x1y1, \
#                         ''  using COL=7 with linespoints	notitle ls 6 axes x1y2, \
#                         ''  using COL=8 with linespoints	notitle ls 7 axes x1y1, \
#                         ''  using COL=9 with linespoints	notitle ls 8 axes x1y2, \
#                         ''  using COL=10 with linespoints	notitle ls 9 axes x1y1, \
#                         ''  using COL=11 with linespoints	notitle ls 10 axes x1y2

# set xlabel "Gaussian Noise SNR (dB)"
# set xrange[0:49]
#set yrange [0:1]
#set y2tics nomirror
#set autoscale y2
#set ylabel "Alexnet Accuracy (%)"
#set ytics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
#set y2label "Energy (J)"
#set ytics nomirror
#set y2tics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
#set output "~/caffe_exp/Paper_Graphs/Output/alex_gvary_SNR.eps"
#plot "~/caffe_exp/Paper_Graphs/Data/for_gnu_alex_gvary_SNR.csv" using COL=2 with linespoints  notitle ls 1 axes x1y1, \
                        # ''  using COL=3 with linespoints        notitle ls 2 axes x1y2, \
                        # ''  using COL=4 with linespoints        notitle ls 3 axes x1y1, \
                        # ''  using COL=5 with linespoints        notitle ls 4 axes x1y2, \
                        # ''  using COL=6 with linespoints        notitle ls 5 axes x1y1, \
                        # ''  using COL=7 with linespoints        notitle ls 6 axes x1y2, \
                        # ''  using COL=8 with linespoints        notitle ls 7 axes x1y1, \
                        # ''  using COL=9 with linespoints        notitle ls 8 axes x1y2, \
                        # ''  using COL=10 with linespoints       notitle ls 9 axes x1y1, \
                        # ''  using COL=11 with linespoints       notitle ls 10 axes x1y2


set xrange[0:49]
set yrange [0:1]
set y2range [0:.0005]
set xlabel "Uniform Noise SNR (dB)"
set ylabel "GoogLeNet Accuracy (%)"
set ytics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set y2tics ("0" 0, "100" .0001, "200" .0002, "300" .0003, "400" .0004, "500" .0005)
set y2label "Energy ({/Symbol m}J)"
set ytics nomirror
set y2tics nomirror
#set autoscale y2
#set y2tics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set output "~/caffe_exp/Paper_Graphs/Output/goog_qvary_SNR.eps"
plot "~/caffe_exp/Paper_Graphs/Data/for_gnu_goog_qvary_SNR.csv" using COL=2 with linespoints  notitle ls 1 axes x1y1, \
                        ''  using COL=3 with linespoints        notitle ls 2 axes x1y2, \
                        ''  using COL=4 with linespoints        notitle ls 3 axes x1y1, \
                        ''  using COL=5 with linespoints        notitle ls 4 axes x1y2, \
                        ''  using COL=6 with linespoints        notitle ls 5 axes x1y1, \
                        ''  using COL=7 with linespoints        notitle ls 6 axes x1y2, \
                        ''  using COL=8 with linespoints        notitle ls 7 axes x1y1, \
                        ''  using COL=9 with linespoints        notitle ls 8 axes x1y2, \
                        ''  using COL=10 with linespoints       notitle ls 9 axes x1y1, \
                        ''  using COL=11 with linespoints       notitle ls 10 axes x1y2, \
                        ''  using COL=12 with linespoints       title "Energy" ls 11 axes x1y1, \
                        ''  using COL=12 with linespoints       title "Top 5 Accuracy" ls 12 axes x1y1, \
                        ''  using COL=12 with linespoints        title "D1" ls 1 axes x1y1, \
                        ''  using COL=12 with linespoints        title "D2" ls 3 axes x1y1, \
                        ''  using COL=12 with linespoints        title "D3" ls 5 axes x1y1, \
                        ''  using COL=12 with linespoints        title "D4" ls 7 axes x1y1, \
                        ''  using COL=12 with linespoints        title "D5" ls 9 axes x1y1, \


set xlabel "Gaussian Noise SNR (dB)"
set xrange[0:49]
set yrange [0:1]
set y2range [0:.0005]
set ylabel "GoogLeNet Accuracy (%)"
set ytics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set y2tics ("0" 0, "100" .0001, "200" .0002, "300" .0003, "400" .0004, "500" .0005)
set y2label "Energy ({/Symbol m}J)"
set ytics nomirror
set y2tics nomirror
#set autoscale y2
#set y2tics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set output "~/caffe_exp/Paper_Graphs/Output/goog_gvary_SNR.eps"
plot "~/caffe_exp/Paper_Graphs/Data/for_gnu_goog_gvary_SNR.csv" using COL=2 with linespoints  notitle ls 1 axes x1y1, \
                        ''  using COL=3 with linespoints        notitle ls 2 axes x1y2, \
                        ''  using COL=4 with linespoints        notitle ls 3 axes x1y1, \
                        ''  using COL=5 with linespoints        notitle ls 4 axes x1y2, \
                        ''  using COL=6 with linespoints        notitle ls 5 axes x1y1, \
                        ''  using COL=7 with linespoints        notitle ls 6 axes x1y2, \
                        ''  using COL=8 with linespoints        notitle ls 7 axes x1y1, \
                        ''  using COL=9 with linespoints        notitle ls 8 axes x1y2, \
                        ''  using COL=10 with linespoints       notitle ls 9 axes x1y1, \
                        ''  using COL=11 with linespoints       notitle ls 10 axes x1y2, \
                        ''  using COL=12 with linespoints       title "Energy" ls 11 axes x1y1, \
                        ''  using COL=12 with linespoints       title "Top 5 Accuracy" ls 12 axes x1y1, \
                        ''  using COL=12 with linespoints        title "D1" ls 1 axes x1y1, \
                        ''  using COL=12 with linespoints        title "D2" ls 3 axes x1y1, \
                        ''  using COL=12 with linespoints        title "D3" ls 5 axes x1y1, \
                        ''  using COL=12 with linespoints        title "D4" ls 7 axes x1y1, \
                        ''  using COL=12 with linespoints        title "D5" ls 9 axes x1y1
