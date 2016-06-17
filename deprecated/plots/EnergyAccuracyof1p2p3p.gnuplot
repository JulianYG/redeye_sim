#What it will eventually look like:
#2 plots total: Energy and Accuracy
#3 points on x-axis: either 1-phase, 2-phase, 3-phase, or high energy, high efficiency, balanced 

reset
set terminal postscript eps color enhanced "Helvetica" 20

color1 = "#B7C951"
color2 = "#6351C9"
color3 = "#FF974D"
color4 = "#d34f4f"
color5 = "#ffee60"

set yrange [0:1]
set xrange [-.4:2.5]
set style data histograms 
set style fill solid 1.0 border lt -1
set datafile separator ","
set tics out nomirror
set xtics ("1" 0, "2" 1, "3" 2)
#set ytics ("0.0" 0, "0.5" 500, "1.0" 1000, "1.5" 1500, "2.0" 2000, "2.5" 2500, "3.0" 3000)

set style fill transparent solid 0.8 border lc rgb '#3D3D3D'
set border 31 front linewidth 2


set xlabel "Phase"
set ylabel "Energy (mJ/J)”


set output "~/caffe_exp/Paper_Graphs/Output/Test_Energy1p2p3p.eps"
plot "~/caffe_exp/Paper_Graphs/Data/mock_data.csv" using COL=1 	title "High Accuracy Condition" lc rgb color1 lw 2, \
                        ''  using COL=2         title "High Energy Condition" lc rgb color2 lw 2, \
                        ''	using COL=3         title "Balanced Condition" lc rgb color3 lw 2


set ylabel "Accuracy (%)”

set output "~/caffe_exp/Paper_Graphs/Output/Test_Accuracy1p2p3p_test.eps"
plot "~/caffe_exp/Paper_Graphs/Data/mock_data.csv" using COL=4 	title "High Accuracy Condition" lc rgb color1 lw 2, \
                        ''  using COL=5 		title "High Energy Condition" lc rgb color2 lw 2, \
                        ''	using COL=6 		title "Balanced Condition" lc rgb color3 lw 2

