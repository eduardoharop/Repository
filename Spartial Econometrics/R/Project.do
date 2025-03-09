clear all
cd "C:\Users\eduar\OneDrive\Escritorio\Masterado\Segundo Semestre\Spatial Econometrics\Project\R files"
spshape2dta realestate2
use realestate2.dta
spmatrix create contiguity W
spgenerate w_medvalsqm2 = W*medvalsqm2

spregress mdvlsq2 imi-unmplym, ml dvarlag(W) ivarlag(W:imi-unmplym) vce(r)

estat impact
estat ic


