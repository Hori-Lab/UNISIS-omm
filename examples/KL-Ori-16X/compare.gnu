set term postscript color 20 lw 2 solid

set out 'compare_sis.ps'
set title 'Epot'
p './md.out' u 0:4 w lp, 'sis_dcd/md.out' u 0:4 w l
set title 'Ebond'
p './md.out' u 0:5 w lp, 'sis_dcd/md.out' u 0:5 w l
set title 'Eangl'
p './md.out' u 0:6 w lp, 'sis_dcd/md.out' u 0:6 w l
set title 'Edih'
p './md.out' u 0:7 w lp, 'sis_dcd/md.out' u 0:7 w l
set title 'Ebp'
p './md.out' u 0:8 w lp, 'sis_dcd/md.out' u 0:8 w l
set title 'Ewca'
p './md.out' u 0:9 w lp, 'sis_dcd/md.out' u 0:9 w l
#set title 'Eele'
#p './md.out' u 0:10 w lp, 'sis_dcd/md.out' u 0:10 w l

set out 'compare_bpcoef.ps'
set title 'Epot'
p './bpcoef_all_src3-1/md.out' u 0:4 w lp, './bpcoef_only_intra_src3-1/md.out' u 0:4 w l
set title 'Ebond'
p './bpcoef_all_src3-1/md.out' u 0:5 w lp, './bpcoef_only_intra_src3-1/md.out' u 0:5 w l
set title 'Eangl'
p './bpcoef_all_src3-1/md.out' u 0:6 w lp, './bpcoef_only_intra_src3-1/md.out' u 0:6 w l
set title 'Edih'
p './bpcoef_all_src3-1/md.out' u 0:7 w lp, './bpcoef_only_intra_src3-1/md.out' u 0:7 w l
set title 'Ebp'
p './bpcoef_all_src3-1/md.out' u 0:8 w lp, './bpcoef_only_intra_src3-1/md.out' u 0:8 w l
set title 'Ewca'
p './bpcoef_all_src3-1/md.out' u 0:9 w lp, './bpcoef_only_intra_src3-1/md.out' u 0:9 w l
#set title 'Eele'
#p './bpcoef_all_src3-1/md.out' u 0:10 w lp, './bpcoef_only_intra_src3-1/md.out' u 0:10 w l

