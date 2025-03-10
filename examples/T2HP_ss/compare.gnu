set term postscript color 20 lw 2 solid
set out 'compare_dcd.ps'
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

set out 'compare_restart.ps'
set title 'Utotal'
p 'run_long/md.out' u 1:4 w lp title 'No restart'\
, 'run1/md.out' u 1:4 w l title 'run1'\
, 'run2_chk/md.out' u 1:4 w l title 'run2 by checkpoint'\
, 'run2_xml/md.out' u 1:4 w l title 'run2 by state xml'
