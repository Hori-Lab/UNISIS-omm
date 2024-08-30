set term postscript color 20 lw 2 solid
set out 'compare.ps'
set title 'Epot'
p 'run1/md.out' u 0:4 w lp, 'sis_dcd/md.out' u 0:4 w l
set title 'Ebond'
p 'run1/md.out' u 0:5 w lp, 'sis_dcd/md.out' u 0:5 w l
set title 'Eangl'
p 'run1/md.out' u 0:6 w lp, 'sis_dcd/md.out' u 0:6 w l
set title 'Edih'
p 'run1/md.out' u 0:7 w lp, 'sis_dcd/md.out' u 0:7 w l
set title 'Ebp'
p 'run1/md.out' u 0:8 w lp, 'sis_dcd/md.out' u 0:8 w l
set title 'Ewca'
p 'run1/md.out' u 0:9 w lp, 'sis_dcd/md.out' u 0:9 w l
set title 'Eele'
p 'run1/md.out' u 0:10 w lp, 'sis_dcd/md.out' u 0:10 w l

