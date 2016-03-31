#CODIGO DE ZVEZDA
# Jaime Forero, Nicolas Rocha

#Importa las librerias necesarias para analizar la imagen
import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#Definimos la funcion que crea la matriz de matrices para ejecutar el algoritmo
def hessiano( matriz ):
        n_side = np.size(matriz[0,:])
        matriz_hessiano = np.zeros((n_side-2,n_side-2,2,2))
        
        matriz_hessiano[0:n_side-2,0:n_side-2,0,0] = matriz[2:n_side,1:n_side-1] -2.0*matriz[1:n_side-1,1:n_side-1] + matriz[0:n_side-2,1:n_side-1]
        matriz_hessiano[0:n_side-2,0:n_side-2,0,1] = (matriz[2:n_side,2:n_side] - matriz[2:n_side, 0:n_side-2] -                                             
                                                       matriz[0:n_side-2, 2:n_side] + matriz[0:n_side-2,0:n_side-2])*0.25
        matriz_hessiano[:,:,1,0] = matriz_hessiano[:,:,0,1]
        matriz_hessiano[0:n_side-2,0:n_side-2,1,1] = matriz[1:n_side-1, 2:n_side] -2.0*matriz[1:n_side-1,1:n_side-1] + matriz[n_side-1, 1:n_side-1]
	return matriz_hessiano

#Definimos la funcion que calcula y devuelve los autovalores de una matriz
def autovalores( matriz_derivadas ):
        n_side = np.size(matriz_derivadas[0,:,0,0])
        autovalores_matriz = np.zeros((n_side, n_side,2))

        #autovalores
        traza = matriz_derivadas[:,:, 0, 0] + matriz_derivadas[:,:,1,1]
        determinante = matriz_derivadas[:,:, 0, 0]*matriz_derivadas[:,:,1,1] - matriz_derivadas[:,:,0,1]*matriz_derivadas[:,:,1,0]
        autovalores_matriz[:,:,0] = 0.5*(traza + np.sqrt(traza**2 - 4.0*determinante))
        autovalores_matriz[:,:,1] = 0.5*(traza - np.sqrt(traza**2 - 4.0*determinante))

	return autovalores_matriz


#Guarda los valores de la imagen en una matriz
hdulist  = fits.open("./data/bbso_tio_pcosr_20130902_162238.fts")
image_data = hdulist[0].data
print ""
print "La imagen ha sido cargada"

plt.imshow( image_data )
plt.show()


#Imprime informacion de la imagen que se cargo
print ""
print "Informacion de la imagen:"
print ""
print hdulist.info()
print ""
print(image_data.shape)
print ""

#cortamos el pedazo que necesitamos
new_image_data = 1.0*image_data[ 1500:-100 , 1500:-100 ]
final_image_data = new_image_data[1:-1,1:-1]


plt.imshow( final_image_data )
plt.show()


#Calculamos la matriz asociada al algoritmo
derivada_matriz = hessiano(new_image_data)
print "La derivada ha sido calculada"

autovalores_matriz = autovalores(derivada_matriz);
print "los autovectores han sido calculados"
print ""


#Graficamos los autovalores bajo ciertas condiciones
graph = np.zeros( ( 441 , 441 ) )
umbral = int(100)
for x in range( 0 , 441 ):
	for y in range( 0 , 441 ):
		if( (autovalores_matriz[ x , y , 1 ] < umbral ) and ( autovalores_matriz[ x , y , 0 ] > umbral ) ):
			graph[x][y] = 1

plt.imshow( graph )
plt.show()


hist_auto1,binsauto1 = np.histogram( autovalores_matriz[:,:,0] , bins = 50 )
hist_auto2,binsauto2 = np.histogram( autovalores_matriz[:,:,1] , bins = 50 )

centros_1 = 0.5*( binsauto1[1:]+binsauto1[0:-1] )
centros_2 = 0.5*( binsauto2[1:]+binsauto2[0:-1] )


plt.plot( centros_1 , np.log10( hist_auto1 + 1 ) )
plt.show()

plt.plot( centros_2 , np.log10( hist_auto2 + 1 ) )
plt.show()


ntot = np.prod(np.shape(autovalores_matriz[:,:,0]))

#histogramas
hist2D, bins2DX , bins2DY= np.histogram2d(np.reshape(autovalores_matriz[:,:,0],ntot) , np.reshape(autovalores_matriz[:,:,1], ntot))

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111)
im = mpl.image.NonUniformImage(ax, interpolation='bilinear')
xcenters = 0.5*( bins2DX[1:] + bins2DX[0:-1] )
ycenters = 0.5*( bins2DY[1:] + bins2DY[0:-1] )
im.set_data(xcenters, ycenters, hist2D)
ax.images.append(im)
ax.set_xlim( bins2DX[0], bins2DX[-1])
ax.set_ylim( bins2DY[0], bins2DY[-1])
ax.set_aspect('equal')
fig.colorbar(im)
plt.show()






