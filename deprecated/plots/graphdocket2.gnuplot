reset
set terminal postscript eps color enhanced "Helvetica" 20

color1 = "#F8E8D8"

color2 = "#9f99a3"
color3 = "#a88b89"
color4 = "#a1b6a4"
color5 = "#a0aeb4"
color6 = "#c4bcb2"

#set yrange [0:1]
set style data histograms 
set style fill solid 1.0 border lt -1
set datafile separator ","
set tics out nomirror
#set ytics ("0.0" 0, "0.5" 500, "1.0" 1000, "1.5" 1500, "2.0" 2000, "2.5" 2500, "3.0" 3000)

set style fill transparent solid 0.8 border lc rgb '#3D3D3D'
set border 31 front linewidth 2

set style line 1 dt 1 lc rgb color1 lw 2 pt 1
set style line 2 lc rgb color2 lw 2
set style line 3 lc rgb color3 lw 2 
set style line 4 lc rgb color4 lw 2 
set style line 5 lc rgb color5 lw 2 
set style line 6 lc rgb color6 lw 2 

set xlabel "Depth"
set ylabel "Energy (J)”
set xrange [-.5:.55]
set yrange [0:2]
set xtics ("Without\n RedEye" -0.39, "1" -.25, "2" -.08, "3" .08, "4" .25, "5" .42)
#set xtics ("Without Anacrusys" 0, "1" 1, "2" 2, "3" 3, "4" 4, "5" 5)
set output "~/caffe_exp/Paper_Graphs/Output/graphdocket2b.eps"
plot "~/caffe_exp/Paper_Graphs/Data/bar_2b.csv" using COL=1 title "Jetson TK1 GPU" lc rgb color1 lw 2, \
                        ''  using COL=7        title "Jetson TK1 CPU" lc rgb color1 fillstyle pattern 4, \
                        ''  using COL=13        notitle ls 2, \
                        ''  using COL=2        notitle ls 2, \
                        ''  using COL=8         notitle lc rgb color2 fillstyle pattern 4, \
                        ''  using COL=13         notitle ls 2, \
                        ''  using COL=3         notitle ls 3, \
                        ''  using COL=9         notitle lc rgb color3 fillstyle pattern 4, \
                        ''  using COL=13         notitle ls 2, \
                        ''  using COL=4        notitle ls 4, \
                        ''  using COL=10         notitle lc rgb color4 fillstyle pattern 4, \
                        ''  using COL=13         notitle ls 2, \
                        ''  using COL=5         notitle ls 5, \
                        ''  using COL=11         notitle lc rgb color5 fillstyle pattern 4, \
                        ''  using COL=13         notitle ls 2, \
                        ''  using COL=6         notitle ls 6, \
                        ''  using COL=12        notitle lc rgb color6 fillstyle pattern 4