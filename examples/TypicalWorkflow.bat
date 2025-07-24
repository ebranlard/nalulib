
gmsh2exo -h # Display help
gmsh2exo diamond_n2.msh -o diamond_n2_gmesh.exo     
exo-info  diamond_n2_gmesh.exo      

plt3d2exo diamond_n2.fmt -o diamond_n2.exo
exo-info  diamond_n2.exo      
plt3d2exo diamond_n2.fmt -o diamond_n1.exo --flatten --loop
exo-layers diamond_n1.exo -n 2000 --layers 
exo-flatten  diamond_n2.exo -o diamond_n1.exo 
exo-zextrude diamond_n1.exo -z 4 -n 20      
exo-rotate   diamond_n20.exo -a 30          
nalu-aseq input.yaml -a -30 30 10 -j polar -b submit.sh

::nalu-restart input.yaml -b submit.sh 
echo "Generated files will be deleted"
pause
rm submit-input*.sh
rm input_aoa*.yaml
rm diamond_*.exo
rm diamond_*.txt
