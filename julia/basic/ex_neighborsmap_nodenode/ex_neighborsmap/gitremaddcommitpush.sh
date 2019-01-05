git rm -rf example-opt example-dbg equation_system_out.e results_* *.mat horz *.png *.o
git rm -rf *.o
rm -rf *.mat relerr* *.png *.e traceout* temp_trace* temp_print* 
rm -rf *.csv *.txt *.e *.cs *.png
git rm -rf core
git rm -rf example-*
git add *.C 
git add *.h
git commit -m '.'
git push
