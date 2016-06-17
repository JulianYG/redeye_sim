#What it will eventually look like:
#2 plots total: Accuracy & Energy vs Gaussian Sigma, Accuracy & Energy vs Uniform Standard Deviation
#x axis represents increasing Gaussian Sigma or increasing Uniform Standard Deviation

reset
set terminal postscript eps color enhanced "Helvetica" 20

color1 = "#B7C951"
color2 = "#6351C9"
color3 = "#FF974D"
color4 = "#d34f4f"
color5 = "#527ACC"

#set style data histograms 
set style fill solid 1.0 border 
set datafile separator ","
set tics out nomirror

set style fill transparent solid 0.8 border lc rgb '#3D3D3D'
set border 31 front linewidth 2

set style line 1 dt 1 lc rgb color1 lw 5 pt 1 pointinterval 5 
set style line 2 dt 3 lc rgb color1 lw 8 pt 1 pointinterval 5
set style line 3 dt 1 lc rgb color2 lw 4 pt 2 pointinterval 5
set style line 4 dt 3 lc rgb color2 lw 7 pt 2 pointinterval 5
set style line 5 dt 1 lc rgb color3 lw 3 pt 3 pointinterval 5
set style line 6 dt 3 lc rgb color3 lw 6 pt 3 pointinterval 5
set style line 7 dt 1 lc rgb color4 lw 2 pt 4 pointinterval 5
set style line 8 dt 3 lc rgb color4 lw 5 pt 4 pointinterval 5
set style line 9 dt 1 lc rgb color5 lw 2 pt 5 pointinterval 5
set style line 10 dt 3 lc rgb color5 lw 4 pt 5 pointinterval 5

set xrange[0:49]
set yrange [0:1]
set y2tics nomirror
set autoscale y2
set ylabel "AlexNet Accuracy (%)"
set xtics("2.0" 0,"3.5" 5,"6.3" 10,"11.4" 15,"20.5" 20,"37.0" 25,"66.5" 30,"120" 35,"215" 40,"387" 45)
set ytics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set y2label "Energy (J)"
set ytics nomirror
#set y2range[0:10e-3]
#set y2tics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set xlabel "Uniform Noise Signal Amplitude (x10^-^3)"
set output "~/caffe_exp/Paper_Graphs/Output/alex_qvary.eps"
plot "~/caffe_exp/Paper_Graphs/Data/for_gnu_alex_qvary.csv" using COL=2 with linespoints title "Top 5 Accuracy, 1 Phase" ls 1 axes x1y1, \
                        ''  using COL=3 with linespoints	title "Energy, 1 Phases" ls 2 axes x1y2, \
                        ''  using COL=4 with linespoints	title "Top 5 Accuracy, 2 Phases" ls 3 axes x1y1, \
                        ''  using COL=5 with linespoints	title "Energy, 2 Phases" ls 4 axes x1y2, \
                        ''  using COL=6 with linespoints	title "Top 5 Accuracy, 3 Phases" ls 5 axes x1y1, \
                        ''  using COL=7 with linespoints	title "Energy, 3 Phases" ls 6 axes x1y2, \
                        ''  using COL=8 with linespoints	title "Top 5 Accuracy, 4 Phases" ls 7 axes x1y1, \
                        ''  using COL=9 with linespoints	title "Energy, 4 Phases" ls 8 axes x1y2, \
                        ''  using COL=10 with linespoints	title "Top 5 Accuracy, 5 Phases" ls 9 axes x1y1, \
                        ''  using COL=11 with linespoints	title "Energy, 5 Phases" ls 10 axes x1y2


set xrange[0:49]
set yrange [0:1]
set ylabel "AlexNet Accuracy (%)"
set ytics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set xtics("1.1" 0, "2.0" 5, "3.7" 10, "6.6" 15, "12.0" 20,  "21.3" 25, "38.4" 30, "69.1" 35,"124" 40,"224" 45)
set y2label "Energy (J)"
set ytics nomirror
set y2tics nomirror
#set autoscale y2
set y2range [0:.09]
set y2tics ("0" 0.0, "" .01, "0.02" 0.02, "" .030, "0.04" .040, "" .050, "0.06" .060, "" .70, "0.08" .080, "" .090)
set xlabel "Gaussian Noise Signal Amplitude (x10^-^3)"
set output "~/caffe_exp/Paper_Graphs/Output/alex_gvary.eps"
plot "~/caffe_exp/Paper_Graphs/Data/for_gnu_alex_gvary.csv" using COL=2 with linespoints  title "Top 5 Accuracy, 1 Phase" lc rgb color1 lw 5 axes x1y1, \
                        ''  using COL=3 with linespoints	title "Energy, 1 Phases" ls 2 axes x1y2, \
                        ''  using COL=4 with linespoints	title "Top 5 Accuracy, 2 Phases" ls 3 axes x1y1, \
                        ''  using COL=5 with linespoints	title "Energy, 2 Phases" ls 4 axes x1y2, \
                        ''  using COL=6 with linespoints	title "Top 5 Accuracy, 3 Phases" ls 5 axes x1y1, \
                        ''  using COL=7 with linespoints	title "Energy, 3 Phases" ls 6 axes x1y2, \
                        ''  using COL=8 with linespoints	title "Top 5 Accuracy, 4 Phases" ls 7 axes x1y1, \
                        ''  using COL=9 with linespoints	title "Energy, 4 Phases" ls 8 axes x1y2, \
                        ''  using COL=10 with linespoints	title "Top 5 Accuracy, 5 Phases" ls 9 axes x1y1, \
                        ''  using COL=11 with linespoints	title "Energy, 5 Phases" ls 10 axes x1y2



#set ylabel "Energy (J)”
set xrange[0:49]
set yrange [0:1]
set ylabel "GoogLeNet Accuracy (%)"
set xtics("2.0" 0,"3.5" 5,"6.3" 10,"11.4" 15,"20.5" 20,"37.0" 25,"66.5" 30,"120" 35,"215" 40,"387" 45)
set ytics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set y2label "Energy (J)"
set ytics nomirror
set y2tics nomirror
set autoscale y2
#set y2tics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set xlabel "Uniform Noise Signal Amplitude (x10^-^3)"
set output "~/caffe_exp/Paper_Graphs/Output/goog_qvary.eps"
plot "~/caffe_exp/Paper_Graphs/Data/for_gnu_goog_qvary.csv" using COL=2 with linespoints  title "Top 5 Accuracy, 1 Phase" lc rgb color1 lw 5 axes x1y1, \
                        ''  using COL=3 with linespoints	title "Energy, 1 Phases" ls 2 axes x1y2, \
                        ''  using COL=4 with linespoints	title "Top 5 Accuracy, 2 Phases" ls 3 axes x1y1, \
                        ''  using COL=5 with linespoints	title "Energy, 2 Phases" ls 4 axes x1y2, \
                        ''  using COL=6 with linespoints	title "Top 5 Accuracy, 3 Phases" ls 5 axes x1y1, \
                        ''  using COL=7 with linespoints	title "Energy, 3 Phases" ls 6 axes x1y2, \
                        ''  using COL=8 with linespoints	title "Top 5 Accuracy, 4 Phases" ls 7 axes x1y1, \
                        ''  using COL=9 with linespoints	title "Energy, 4 Phases" ls 8 axes x1y2, \
                        ''  using COL=10 with linespoints	title "Top 5 Accuracy, 5 Phases" ls 9 axes x1y1, \
                        ''  using COL=11 with linespoints	title "Energy, 5 Phases" ls 10 axes x1y2



set xrange[0:49]
set yrange [0:1]
set ylabel "GoogLeNet Accuracy (%)"
set ytics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set xtics("1.1" 0, "2.0" 5, "3.7" 10, "6.6" 15, "12.0" 20,  "21.3" 25, "38.4" 30, "69.1" 35,"124" 40,"224" 45)
set y2label "Energy (J)"
set ytics nomirror
set y2tics nomirror
set y2range [0:.09]
set y2tics ("0" 0.0, "" .01, "0.02" 0.02, "" .030, "0.04" .040, "" .050, "0.06" .060, "" .70, "0.08" .080, "" .090)
#set y2tics ("0" 0, "" .10, "20" .20, "" .30, "40" .40, "" .50, "60" .60, "" .70, "80" .80, "" .90, "100" 1.00)
set xlabel "Gaussian Noise Signal Amplitude (x10^-^3)"
set output "~/caffe_exp/Paper_Graphs/Output/goog_gvary.eps"
plot "~/caffe_exp/Paper_Graphs/Data/for_gnu_goog_gvary.csv" using COL=2 with linespoints  title "Top 5 Accuracy, 1 Phase" lc rgb color1 lw 5 axes x1y1, \
                        ''  using COL=3 with linespoints	title "Energy, 1 Phases" ls 2 axes x1y2, \
                        ''  using COL=4 with linespoints	title "Top 5 Accuracy, 2 Phases" ls 3 axes x1y1, \
                        ''  using COL=5 with linespoints	title "Energy, 2 Phases" ls 4 axes x1y2, \
                        ''  using COL=6 with linespoints	title "Top 5 Accuracy, 3 Phases" ls 5 axes x1y1, \
                        ''  using COL=7 with linespoints	title "Energy, 3 Phases" ls 6 axes x1y2, \
                        ''  using COL=8 with linespoints	title "Top 5 Accuracy, 4 Phases" ls 7 axes x1y1, \
                        ''  using COL=9 with linespoints	title "Energy, 4 Phases" ls 8 axes x1y2, \
                        ''  using COL=10 with linespoints	title "Top 5 Accuracy, 5 Phases" ls 9 axes x1y1, \
                        ''  using COL=11 with linespoints	title "Energy, 5 Phases" ls 10 axes x1y2




