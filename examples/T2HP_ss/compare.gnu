set term postscript color 20 lw 2 solid
set out 'compare.ps'
set title 'Utotal'
p 'run1/md.out' u 0:4 w lp, 'sis_dcd/md.out' u 0:4 w lp, 'omm_dcd/md.out' u 0:2 w l
set title 'Ubond'
p 'run1/md.out' u 0:5 w lp, 'sis_dcd/md.out' u 0:5 w lp, 'omm_dcd/md.out' u 0:3 w l
set title 'Uangl'
p 'run1/md.out' u 0:6 w lp, 'sis_dcd/md.out' u 0:6 w lp, 'omm_dcd/md.out' u 0:4 w l
set title 'Udih'
p 'run1/md.out' u 0:7 w lp, 'sis_dcd/md.out' u 0:7 w lp, 'omm_dcd/md.out' u 0:5 w l
set title 'Ubp'
p 'run1/md.out' u 0:8 w lp, 'sis_dcd/md.out' u 0:8 w lp, 'omm_dcd/md.out' u 0:6 w l
set title 'Uwca'
p 'run1/md.out' u 0:9 w lp, 'sis_dcd/md.out' u 0:9 w lp, 'omm_dcd/md.out' u 0:7 w l
set title 'Uele'
p 'run1/md.out' u 0:10 w lp, 'sis_dcd/md.out' u 0:10 w lp, 'omm_dcd/md.out' u 0:8 w l

