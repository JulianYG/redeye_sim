reset
set terminal postscript eps color enhanced "Helvetica" 20
set size 0.55,1.2

color1 = "#F8E8D8"

color2 = "#9f99a3"
color3 = "#a88b89"
color4 = "#a1b6a4"
color5 = "#a0aeb4"
color6 = "#c4bcb2"

color7 = "#808080"

set key samplen 1.5
set key spacing 4.5
set key at .45, 1e7

#set yrange [0:1]
set style data histograms 
#set boxwidth .5
set style fill pattern 1.0 border lt -1
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
set style line 7 lc rgb color7 lw 2 

set xlabel "Depth"
set ylabel "Energy ({/Symbol m}J)”
set logscale y
set xrange [-.48:.43]
set xtics ("IS" -0.38, "1" -.25, "2" -.12, "3" .03, "4" .18, "5" .31)
set output "~/caffe_exp/Paper_Graphs/Output/graphdocket1a.eps"
plot "~/caffe_exp/Paper_Graphs/Data/bar_1a_new.csv" using COL=11 title "Image Sensor Energy" fillstyle ls 1, \
                        ''  using COL=12        notitle ls 2, \
                        ''  using COL=1         notitle ls 2, \
                        ''  using COL=6         notitle lc rgb color2 fillstyle pattern 4, \
                        ''  using COL=12         notitle ls 2, \
                        ''  using COL=2         notitle ls 3, \
                        ''  using COL=7         notitle lc rgb color3 fillstyle pattern 4, \
                        ''  using COL=12         notitle ls 2, \
                        ''  using COL=3        notitle ls 4, \
                        ''  using COL=8         notitle lc rgb color4 fillstyle pattern 4, \
                        ''  using COL=12         notitle ls 2, \
                        ''  using COL=4         notitle ls 5, \
                        ''  using COL=9         notitle lc rgb color5 fillstyle pattern 4, \
                        ''  using COL=12         notitle ls 2, \
                        ''  using COL=5         notitle ls 6, \
                        ''  using COL=10        notitle lc rgb color6 fillstyle pattern 4, \
                        ''  using COL=13         title "Quant. Energy\n at SNR=26" ls 7, \
                        ''  using COL=14        title "Comp. Energy\n at SNR=40" lc rgb color6 fillstyle pattern 4, \


unset logscale y
set xrange [-.48:.55]
set xlabel "Depth"
set ylabel "Seconds”
set yrange[0:.4]
set ytics ("0" 0, "0.1" .1, "0.2" .2, "0.3" .3, "0.4" .4)
set xtics ("IS" -0.34, "1" -.19, "2" -.04, "3" .115, "4" .265, "5" .42)

set output "~/caffe_exp/Paper_Graphs/Output/graphdocket1b.eps"
plot "~/caffe_exp/Paper_Graphs/Data/bar_1b.csv" using COL=1 notitle ls 1, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=2         notitle ls 2, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=3         notitle ls 3, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=4         notitle ls 4, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=5         notitle ls 5, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=6         notitle ls 6, \

set yrange[0:2]
set ytics ("0.0" 0, "0.5" .5, "1.0" 1, "1.5" 1.5, "2.0" 2.0)
set xlabel "Depth"
set ylabel "Mbits/frame"
set xtics ("IS" -0.34, "1" -.19, "2" -.04, "3" .115, "4" .265, "5" .42)

set output "~/caffe_exp/Paper_Graphs/Output/graphdocket1c.eps"
plot "~/caffe_exp/Paper_Graphs/Data/bar_1c.csv" using COL=1 notitle ls 1, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=2         notitle ls 2, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=3         notitle ls 3, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=4         notitle ls 4, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=5         notitle ls 5, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=6         notitle ls 6, \

set yrange[0:32]
set ytics ("0.0" 0, "5" 5, "10" 10, "15" 15, "20" 20, "25" 25, "30" 30)
set xlabel "Depth"
set ylabel "Bitrate (Mbps)"
set xtics ("IS" -0.34, "1" -.19, "2" -.04, "3" .115, "4" .265, "5" .42)

set output "~/caffe_exp/Paper_Graphs/Output/graphdocket1d.eps"
plot "~/caffe_exp/Paper_Graphs/Data/bar_1d.csv" using COL=1 notitle ls 1, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=2         notitle ls 2, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=3         notitle ls 3, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=4         notitle ls 4, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=5         notitle ls 5, \
                        ''  using COL=7         notitle ls 2, \
                        ''  using COL=6         notitle ls 6, \

