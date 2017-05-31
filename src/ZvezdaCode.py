#CODIGO DE ZVEZDA
# Jaime Forero, Nicolas Rocha

#Importa las librerias necesarias para analizar la imagen
import numpy as np
from astropy.io import fits
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.widgets import Slider,Button



#Definimos la funcion que crea la matriz de matrices para ejecutar el algoritmo
def hessiano( matriz ):
        n_side = np.size(matriz[0,:])
        matriz_hessiano = np.zeros((n_side-2,n_side-2,2,2))
        
        matriz_hessiano[0:n_side-2,0:n_side-2,0,0] = matriz[2:n_side,1:n_side-1] -2.0*matriz[1:n_side-1,1:n_side-1] + matriz[0:n_side-2,1:n_side-1]
        matriz_hessiano[0:n_side-2,0:n_side-2,0,1] = (matriz[2:n_side,2:n_side] - matriz[2:n_side, 0:n_side-2] - matriz[0:n_side-2, 2:n_side] + matriz[0:n_side-2,0:n_side-2])*0.25
        matriz_hessiano[:,:,1,0] = matriz_hessiano[:,:,0,1]
        matriz_hessiano[0:n_side-2,0:n_side-2,1,1] = matriz[1:n_side-1, 2:n_side] -2.0*matriz[1:n_side-1,1:n_side-1] + matriz[n_side-1, 1:n_side-1]
        return (matriz_hessiano)
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
hdulist  = fits.open("../data/bbso_tio_pcosr_20120726_171242.fts")
image_data = hdulist[0].data
print ("")
print ("La imagen ha sido cargada")


#Imprime informacion de la imagen que se cargo
print ("")
print ("Informacion de la imagen:")
print ("")
print (hdulist.info())
print ("")
print(image_data.shape)
print ("")

#cortamos el pedazo que necesitamos
new_image_data = 1.0*image_data[ 500:-500, 500:-500 ]
final_image_data = new_image_data[1:-1,1:-1]


#Calculamos la matriz asociada al algoritmo
derivada_matriz = hessiano(new_image_data)
print ("La derivada ha sido calculada")


autovalores_matriz = autovalores(derivada_matriz);
print ("los autovectores han sido calculados")
print ("")

plt.figure(figsize=(15,10))
plt.subplot(131)
plt.imshow( final_image_data , cmap='gray' )
plt.title("Image to be analyzed")
plt.colorbar()
plt.subplot(132)
plt.imshow( autovalores_matriz[:,:,0] , cmap='gray' )
plt.title( "First eigenvalues" )
plt.colorbar()
plt.subplot(133)
plt.imshow( autovalores_matriz[:,:,1] , cmap='gray' )
plt.title( "Second eigenvalues" )
plt.colorbar()
plt.show()



dimension = len( autovalores_matriz )
print ("Dimension de los autovalores: " + str(dimension))



b=100
des=np.std(autovalores_matriz[:,:,0])
print ('Desviacion estandar datos es '+str(des))



def umbra(umbral):

        #Graficamos los autovalores bajo ciertas condiciones
        graph_positive = np.zeros( ( dimension , dimension ) )
        graph_negative = np.zeros( ( dimension , dimension ) )
        graph_diffsp = np.zeros( ( dimension , dimension ) )

        
        for x in range( 0 , dimension ):
	        for y in range( 0 , dimension ):
		        if( ( autovalores_matriz[ x , y , 1 ] > umbral ) and ( autovalores_matriz[ x , y , 0 ] > umbral ) ):
			        graph_positive[ x , y ] = 1

		        if( ( autovalores_matriz[ x , y , 1 ] < umbral ) and ( autovalores_matriz[ x , y , 0 ] < umbral ) ):
			        graph_negative[ x , y ] = 1

		        if( ( autovalores_matriz[ x , y , 1 ] < umbral ) and ( autovalores_matriz[ x , y , 0 ] > umbral ) ):
			        graph_diffsp[ x , y ] = 1

        return graph_positive,graph_negative,graph_diffsp



fig, ax=plt.subplots()
plt.subplots_adjust(left=0.15, bottom=0.25)

umbral = des*3./4.
ui = des/4.
uf = des*5./4.


gp,gn,gd=umbra(umbral)

plt.subplot(221)
g1=plt.imshow( autovalores_matriz[:,:,0] , cmap='gray' )
plt.title("First eingevalues")

plt.subplot(222)
g2=plt.imshow( gn , cmap='gray' )
plt.title("$\lambda_{1}$ < $\lambda_{u}$ y $\lambda_{2}$ < $\lambda_{u}$")

plt.subplot(223)
g3=plt.imshow( gd , cmap='gray' )
plt.title("$\lambda_{1}$ > $\lambda_{u}$ y $\lambda_{2}$ < $\lambda_{u}$")

plt.subplot(224)
g4=plt.imshow( final_image_data , cmap='gray' )
plt.title("imagen")

axlu = plt.axes([0.25, 0.1, 0.65, 0.03])
slu=Slider(axlu,'$\lambda_{u}$',ui,uf,valinit=umbral)


def update(val):
        lu=slu.val
        g_p,g_n,g_d=umbra(lu)
        g2.set_data(g_n)
        g3.set_data(g_d)
        fig.canvas.draw_idle()
                
slu.on_changed(update)

saveax=plt.axes([0.8,0.025,0.1,0.04])
button=Button(saveax,'Save')

def Save(event):
        nombre=str(input("Nombre de la imagen: "))
        extent = g3.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(nombre+'.pdf', bbox_inches=extent.expanded(1., 1.))
        ex = g4.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        fig.savefig(nombre+'original.pdf', bbox_inches=ex.expanded(1., 1.))
        print ("imagen ha sido guardada")

        
button.on_clicked(Save)     
plt.show()

