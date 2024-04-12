from tkinter import * 
from tkinter import messagebox
from tkinter import PhotoImage
from PIL import Image, ImageTk , ImageSequence
import random
import time
from tkinter import ttk
import tkinter as tk
import os
import webbrowser


#from autoencoder_final import load_autoencoder_model

#Pour les path j'ai mit ceux connecté à mon Pc , faudra juste faire attention mais j'ai tout mit dans le git les images
#J'ai mit sur github  dans le document template_doc , les templates 

#j'ai fait de mon mieux pour expliquer si y'a des problèmes hésite pas , je sens que la connection au back va etre chiant !!
#je travaillerai dessus dans la semaine.


class Logicielprincipal(tk.Frame) : 
    def __init__(self,master=None) : 
        super().__init__(master)
        self.master = master
        self.pack()
        self.frame_select = [] # les photos sélectionnées pour l'algo génétique
        self.frame_confirm = [] # les photos sélectionnées pour le tueur 
        self.red = True # savoir le mode  rouge = algo gene , vert = tueur
        self.bool_finish = False # ce bool permet de savoir si on peut cliquer sur finish
        self.number_image = 0 # permet de savoir combien d'image a sélectionné l'utilisateur
        self.nouvelle_fenetre = self.menu_entree() # on charge le menu d'entrée d'abord
        self.menu_deroulant = self.MenuDeroulant(master=self.nouvelle_fenetre)
        self.autoencoder_liste_box_A = self.List_box(master = self.nouvelle_fenetre)
        self.liste_box_genetic = self.choice_genetic(master = self.nouvelle_fenetre)
        self.choix_nb_image_var = self.choix_nb_image(master = self.nouvelle_fenetre)
        self.model = ""
        self.genetic_value = 0


    def menu_sortie(self) :  #fonction pour enlever la fenêtre Menu 
        self.number_image = self.choix_nb_image_var.var.get() #on récupère le nombre d'image qui a été choisi
        if self.number_image != 0 : # on vérifie que l'utilisateur a choisi un nb de photo 
            #copie du nom du modèle 
            selected_value = self.autoencoder_liste_box_A.selected_value
            self.genetic_value = self.liste_box_genetic.genetic_value
            self.model = "graphique/model/" + str(selected_value)
            print("ici 2: " , self.model)
            print("ici : " , self.genetic_value)
            self.nouvelle_fenetre.destroy()
            self.interface_object() # on charge la fenetre principal 
            self.master.deiconify()
       
    def menu_entree(self) : #Ceci est les composants de notre Menu

        #initialise le nouveau menu 
        self.master.withdraw() # permet de cacher la fenêtre principal
        nouvelle_fenetre_menu = Toplevel(self.master)
        nouvelle_fenetre_menu.title("Menu")
        nouvelle_fenetre_menu.geometry("900x900")
        nouvelle_fenetre_menu.configure(bg="light blue")

        #permet de charger l'image de fond du menu 

        photo_background = Image.open("C:/Users/lukyl/Music/javascript/__pycache__/graphique/template_doc/Entry_Menu.png") # on ouvre l'image avec ¨PIL
        photo_background = photo_background.resize((900,200) , Image.LANCZOS) # on resize l'image
        photo_background = ImageTk.PhotoImage(photo_background) # on l'a convertit en image tkinter
        Label_background = Label(nouvelle_fenetre_menu , image = photo_background)
        Label_background.image = photo_background #obligé sinon tkinter ne stock pas l'image
        Label_background.place(x=0,y=0)
        

        #bouton de fin de menu 
        nouvelle_fenetre_button = Button(nouvelle_fenetre_menu , text = "Logiciel principal" , command = self.menu_sortie)
        nouvelle_fenetre_button.place( x= 390 , y = 550)

        return nouvelle_fenetre_menu

        #ici est un concept pour un menu déroulant (pas important) , je le mettrai plus sur le vrai logiciel 
            
    class MenuDeroulant(tk.Frame) : 
        def __init__(self,master=None) : 
            super().__init__(master)
            self.master = master
            self.pack()
            self.create_widgets()
            self.tutoriel_frame_liste = ["C:/Users/lukyl/OneDrive/Images/giphy.gif" , 
                                         "C:/Users/lukyl/OneDrive/Images/giphy_1.gif",
                                         "C:/Users/lukyl/OneDrive/Images/giphy.gif",
                                         "C:/Users/lukyl/OneDrive/Images/giphy_1.gif",]
            self.index_tutoriel = 0
            
        def create_widgets(self) : 

            #création de la zone de contenu caché 
            self.menu_contenu = Frame(self.master,bg = "white" , width = 200)
            self.menu_contenu.pack_propagate(False)
            self.menu_contenu.place(x=-200,y=0,relheight=1,relwidth=0.2)

            #création de la zone bouton pour affciher le menu 

            self.bouton_menu = Button(self.master,text="Help",command=self.toggle_menu)
            self.bouton_menu.place(x=10,y=10)

            self.label_menu = Label(self.menu_contenu,text="Liste help" , bg= "lightgrey") 
            self.label_menu.pack(pady=10)
            
            self.items1 = Button(self.menu_contenu,text= "tutoriel" , command = self.init_tutoriel)
            self.items1.pack(pady=5)

            self.items2 = Button(self.menu_contenu,text = "lien web" , command = self.open_website_tutoriel)
            self.items2.pack(pady = 5)

            

        def move_contenu(self) :
                    step = 1
                    position_x = self.menu_contenu.winfo_x() 
                    if position_x != 0 : 
                        self.menu_contenu.place(x= position_x + step)
                        if (position_x + step) == 0 : 
                            return
                        
                        self.master.after(2,self.toggle_menu)      

        def toggle_menu(self) : 
           
            if self.menu_contenu.winfo_x() < 0 : 
                self.move_contenu()    
            else : 
                self.menu_contenu.place(x=-200)

        
        def init_tutoriel(self) : 

            tutoriel_fenetre = Toplevel(self.master)
            tutoriel_fenetre.title("tutoriel")
            tutoriel_fenetre.geometry("600x600")
            tutoriel_fenetre.configure(bg="light blue")

            nouvelle_image = Image.open(self.tutoriel_frame_liste[0])
            nouvelle_image  = ImageTk.PhotoImage(nouvelle_image)

            self.Frame_tutoriel = Frame(tutoriel_fenetre,width=500 , height = 500)
            self.Frame_tutoriel.place(x=50,y=50)
            self.label_tutoriel = Label(self.Frame_tutoriel , image = nouvelle_image)
            self.label_tutoriel.image = nouvelle_image
            self.label_tutoriel.place(x=0,y=0)
         

            self.Button_suivant = Button(tutoriel_fenetre,text = "suivant" , bg = "pink" , width = 15 , command = lambda  : self.change_frame("suivant"))
            self.Button_suivant.place(x = 400 , y = 575)

            self.Button_retour = Button(tutoriel_fenetre , text =  "retour" , bg = "lightgrey" , width = 15 , command = lambda : self.change_frame("retour"))
            self.Button_retour.place(x = 100 , y = 575)

            self.number_label_index = Label(tutoriel_fenetre , text = "0" , width = 10 , bg = "white")
            self.number_label_index.place(x = 275 , y = 575)

        def change_frame(self,method) :
            if method == "retour" : 
                print("test2")
                if self.index_tutoriel >= 1 : 
                    self.index_tutoriel-= 1
                    nouvelle_image = Image.open(self.tutoriel_frame_liste[self.index_tutoriel])
                    nouvelle_image  = ImageTk.PhotoImage(nouvelle_image)
                    self.label_tutoriel.config(image = nouvelle_image)
                    self.label_tutoriel.image = nouvelle_image
                    self.label_tutoriel.place(x=0,y=0)
                    self.check_state()
            if method == "suivant" : 
                if self.index_tutoriel < len(self.tutoriel_frame_liste)-1 : 
                    self.index_tutoriel+= 1
                    nouvelle_image = Image.open(self.tutoriel_frame_liste[self.index_tutoriel])
                    nouvelle_image  = ImageTk.PhotoImage(nouvelle_image)
                    self.label_tutoriel.config(image = nouvelle_image)
                    self.label_tutoriel.image = nouvelle_image
                    self.label_tutoriel.place(x=0,y=0)
                    self.check_state()

        def check_state(self) : 
          
            if self.index_tutoriel == 0 : 
                self.Button_retour.config(bg = "lightgrey")
            else : 
                self.Button_retour.config(bg = "pink")

    
            if self.index_tutoriel == len(self.tutoriel_frame_liste)-1 : 
                self.Button_suivant.config(bg = "lightgrey")
            else : 
                self.Button_suivant.config(bg = "pink")
            self.number_label_index.config(text=f"{self.index_tutoriel}")
        def open_website_tutoriel(self) : 
            webbrowser.open("https://www.youtube.com/")
          
            
    #code qui s'occupe de la case pour le choix de l'algo 
    class List_box(tk.Frame) : 
        def __init__(self,master=None) : 
            super().__init__(master)
            self.master = master
            self.pack()
            self.selected_value = ""
            self.create_widgets()
            self.list_box_A.bind("<<ListboxSelect>>", lambda event  : self.update_label())

        def create_widgets(self) : 
            
            liste_box_A_frame = Frame(self.master,bg = "White" , width=500 , height = 100) # frame blanc
            self.list_box_A = Listbox(self.master , height = 5) # la liste box
            self.choix_selection = Label(self.master , text = "Aucun" , width=20,height=2 , bg = "White") # Le texte pour le choix algo
            self.choix_selection.place(x= 500  , y= 250)
            liste_box_A_frame.place(x=200,y=230)
            self.list_box_A.place(x=200,y=235)
            #Le code ici permet de récupérer dans le dossier model , le nom des models
            path_model = "C:/Users/lukyl/Music/javascript/__pycache__/graphique/model/"
            if os.path.isdir(path_model) : 
                models = os.listdir(path_model)
                for i in range(len(models))  : 
                    self.list_box_A.insert(END,models[i])

            # permet d'actualiser le nom du label quand on choisi un model dans notre liste box 

        def update_label(self) : 
            selected_index = self.list_box_A.curselection()
            if selected_index : 
                selected_value = self.list_box_A.get(selected_index[0])
                self.choix_selection.config(text=selected_value)
                self.selected_value = selected_value

        

        # même principe que la liste box pour l'algo
            
    class choice_genetic(tk.Frame) : 
        def __init__(self,master=None) : 
            super().__init__(master)
            self.master = master
            self.pack()
            self.genetic_value = 0
            self.create_widgets()
            self.list_box_G.bind("<<ListboxSelect>>", lambda event  : self.update_label())
           

        def create_widgets(self) : 
            liste_box_G_frame = Frame(self.master,bg = "White" , width=500 , height = 100)
            self.choix_selection_G = Label(self.master , text = "Aucun" , width=20,height=2 , bg = "White")
            self.choix_selection_G.place(x= 500  , y= 380)

            liste_box_G_frame.place(x=200 , y= 350)
            self.list_box_G = Listbox(self.master , height = 5)
            self.list_box_G.place(x=200,y=360)
            for item in ["genetic_1" , "genetic_2" , "genetic_3" ] : 
                self.list_box_G.insert(END,item)

        def update_label(self) : 
            selected_index = self.list_box_G.curselection()
            if selected_index : 
                selected_value = self.list_box_G.get(selected_index[0])
                self.genetic_value = selected_value
                self.choix_selection_G.config(text=selected_value)

        
   
            
    class choix_nb_image(tk.Frame) : # ici la fonction pour choisir le nombre d'image
        def __init__(self,master=None) : 
            super().__init__(master)
            self.master = master
            self.pack()
            self.var = IntVar()
            self.var.set(0)
            self.create_widgets()
        
        def create_widgets(self) : 
            button_4 = Radiobutton(self.master,text="4 images",variable =self.var , value = 4)
            button_8 = Radiobutton(self.master,text = "9 images", variable = self.var , value = 9)
            button_12 = Radiobutton(self.master,text = "16 images" , variable = self.var , value = 16)

            button_4.place(x = 400 , y = 470)
            button_8.place(x = 400 , y = 490)
            button_12.place(x = 400 ,  y = 510)
   

    def interface_object(self) : # Ceci est les composants de notre interface principal
        #préparation de l'image pour le background du logiciel 
        print("ici : " , self.model)
        background_image_1 = Image.open("C:/Users/lukyl/OneDrive/Images/vector_background.jpg")
        background_image_1 = background_image_1.resize((1200,800) , Image.LANCZOS)
        background_image_2 = ImageTk.PhotoImage(background_image_1)
        
        # On place l'image n et on définit la taille de la fenêtre en fonction du background de l'image
        background_label = Label(self.master,image = background_image_2)
        background_label.image = background_image_2
        background_label.place(x=0,y=0,relwidth=1,relheight=1)
        self.master.geometry(f"{background_image_1.width}x{background_image_1.height}")

        #carré bleu derrières les photos
        frame_support_centre  = Frame(self.master,width=400,height = 550 , background="light blue")
        frame_support_centre.place(x=82, y = 80)

        #carré bleu derrière la photo du meurtrier
        frame_fond_meurtrier = Frame(self.master,width= 200 , height = 330 , background = "light blue")
        frame_fond_meurtrier.place(x=936 , y = 230)

        #carré bleu pour lier le fond des images et du meurtrier
        frame_centre_ligne = Frame(self.master,width=750,height = 100 , background = "light blue")
        frame_centre_ligne.place(x=350 , y = 300)

        #chargement de l'image de la flèche 
        fleche_image = Image.open("C:/Users/lukyl/OneDrive/Images/fleche_bleu_t.png")
        fleche_image_size = fleche_image.resize((200,95) , Image.LANCZOS)
        fleche_image_tk = ImageTk.PhotoImage(fleche_image_size)

        #placement des flèches dans le logiciel 
        test_label = Label(self.master,image = fleche_image_tk , background="light blue")
        test_label.image = fleche_image_tk
        test_label.place(x = 500 , y = 300)
        test_label_copy = Label(self.master,image = fleche_image_tk , background = "light blue")
        test_label_copy.image = fleche_image_tk
        test_label_copy.place(x=700, y = 300)

        """Ici on a le code pour avoir le nombre de photo qu'on a sélectionné au départ ! """

        if self.number_image == 4 : 
            self.resize = [] # c'est vide car on a pas besoin de resize les photo si on a 4 images
            #Code ici permet de définir les 4 Frames dans l'espace de notre logiciel 
            HLT = 8 #HLT correspond à l'épaisseur en pixel de la bordure de la marge (permmettant la coloration rouge ou vert)
            self.frame1 = Frame(self.master, bd=0, width=178 + HLT, height=218+HLT)  # Spécifier la taille du frame
            self.frame2 = Frame(self.master, bd=0, width=178+HLT, height=218+HLT)
            self.frame3 = Frame(self.master, bd=0, width=178+HLT, height=218+HLT)
            self.frame4 = Frame(self.master, bd=0, width=178+HLT, height=218+HLT)

            #Code ici permet de placer les 4 Frames dans l'espace
            self.frame1.place(x=100-HLT, y=100-HLT)  # Spécifiez les coordonnées en pixels
            self.frame2.place(x=100+178+HLT, y=100-HLT)
            self.frame3.place(x=100-HLT, y=100+218+HLT)
            self.frame4.place(x=100+178+HLT, y=100+218+HLT)

            self.liste_frame = [[self.frame1],[self.frame2],[self.frame3],[self.frame4]] #on stock les frames existant 

            """ J'ai choisi ce type de structure de donnée car j'avais besoin d'alléger le load de image dans chaque
            frame donc j'ai fait une liste contenant chaque frame 
            ATTENTION : quand on load les images chaque Frame a sa propre liste avec en position 0  : le Frame et en 
            position 1 : le path de l'image qui a été mit dans le Frame

            J'en avais besoin pour récupérer l'image du tueur et la resize ( quand tu cliques sur confirmer , ça déplace 
            l'image mais l'image chargé  dans les cases est petite si on a 9/16 images et on peut pas resize les images 
            en récupérant simplement depuis le label. On est obligé d'avoir le path de l'image pour resizer)
            C'est pourquoi y'a cette structure de donnée un peu complexe
            
            """
        


            i = 0 
            for item in self.liste_frame :  # on charge les images dans chaque frame
                 self.load_and_display_image(self.choice_path_image() , item[0] , self.resize , i )
                 i+=1
                
        if self.number_image == 9 : 
            self.resize = [100,120]
            x = 115
            y = 110
            HLT = 8 # HLT correspond à la bordure de marge ( HightLightThickeness , pour avoir une bordure qui apparait bien
            #dans notre frame , il faut laisser de la place à la bordure et donc rogner légèrement les images
            pas_droite = 120 # ça déplace de 120 pixels vers la droite entre 2 images
            pas_en_bas = 150
            self.frame1 = Frame(self.master, bd=0, width=100 + HLT, height=120+HLT , background="black")
            self.frame2 = Frame(self.master, bd=0, width=100 + HLT, height=120+HLT , background="black")
            self.frame3 = Frame(self.master, bd=0, width=100 + HLT, height=120+HLT , background="black")

            self.frame4 = Frame(self.master, bd=0, width=100 + HLT, height=120+HLT , background="black")
            self.frame5 = Frame(self.master, bd=0, width=100 + HLT, height=120+HLT , background="black")
            self.frame6 = Frame(self.master, bd=0, width=100 + HLT, height=120+HLT , background="black")

            self.frame7 = Frame(self.master, bd=0, width=100 + HLT, height=120+HLT , background="black")
            self.frame8 = Frame(self.master, bd=0, width=100 + HLT, height=120+HLT , background="black")
            self.frame9 = Frame(self.master, bd=0, width=100 + HLT, height=120+HLT , background="black")

            self.frame1.place(x = x-HLT, y= y-HLT)
            self.frame2.place(x = x+ pas_droite-HLT, y=y-HLT)
            self.frame3.place(x = x+2*pas_droite-HLT, y=y-HLT)

            self.frame4.place(x = x-HLT, y=y-HLT + pas_en_bas)
            self.frame5.place(x = x+pas_droite-HLT, y=y-HLT + pas_en_bas)
            self.frame6.place(x = x+2*pas_droite-HLT, y=y-HLT + pas_en_bas)

            self.frame7.place(x = x+-HLT, y=y-HLT + 2*pas_en_bas)
            self.frame8.place(x = x+pas_droite-HLT, y=y-HLT + 2*pas_en_bas)
            self.frame9.place(x = x+2*pas_droite-HLT, y=y-HLT + 2*pas_en_bas)

             #Des images vont être chargé dans chaque Frame 
            self.liste_frame = [[self.frame1],[self.frame2],[self.frame3],[self.frame4],[self.frame5],[self.frame6],[self.frame7]
                                ,[self.frame8],[self.frame9]]
            i = 0 # permet de savoir à quel itération du frame , on est c'est pour stocker 
            for item in self.liste_frame : 
                self.load_and_display_image(self.choice_path_image() , item[0] , self.resize , i)
                i+=1
        
        if self.number_image == 16 : ####!!!!!!!!!!
            self.resize = [80,100]
            x = 95
            y = 95
            HLT = 8 
            pas_droite = 100
            pas_en_bas = 120
            self.frame1 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")
            self.frame2 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")
            self.frame3 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")
            self.frame4 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")

            self.frame5 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")
            self.frame6 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")
            self.frame7 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")
            self.frame8 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")

            self.frame9 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")
            self.frame10 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")
            self.frame11= Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")
            self.frame12 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")

            self.frame13 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")
            self.frame14 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")
            self.frame15= Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")
            self.frame16 = Frame(self.master, bd=0, width=80 + HLT, height=100+HLT , background="black")

            self.frame1.place(x = x-HLT, y= y-HLT)
            self.frame2.place(x = x+ pas_droite-HLT, y=y-HLT)
            self.frame3.place(x = x+2*pas_droite-HLT, y=y-HLT)
            self.frame4.place(x = x+3*pas_droite-HLT, y=y-HLT)

            self.frame5.place(x = x-HLT, y= y-HLT+pas_en_bas)
            self.frame6.place(x = x+ pas_droite-HLT, y=y-HLT+pas_en_bas)
            self.frame7.place(x = x+2*pas_droite-HLT, y=y-HLT+pas_en_bas)
            self.frame8.place(x = x+3*pas_droite-HLT, y=y-HLT+pas_en_bas)

            self.frame9.place(x = x-HLT, y= y-HLT+2*pas_en_bas)
            self.frame10.place(x = x+ pas_droite-HLT, y=y-HLT+2*pas_en_bas)
            self.frame11.place(x = x+2*pas_droite-HLT, y=y-HLT+2*pas_en_bas)
            self.frame12.place(x = x+3*pas_droite-HLT, y=y-HLT+2*pas_en_bas)

            self.frame13.place(x = x-HLT, y= y-HLT+3*pas_en_bas)
            self.frame14.place(x = x+ pas_droite-HLT, y=y-HLT+3*pas_en_bas)
            self.frame15.place(x = x+2*pas_droite-HLT, y=y-HLT+3*pas_en_bas)
            self.frame16.place(x = x+3*pas_droite-HLT, y=y-HLT+3*pas_en_bas)

            #Des images vont être chargé dans chaque Frame 
            self.liste_frame = [[self.frame1],[self.frame2],[self.frame3],[self.frame4],[self.frame5],[self.frame6],[self.frame7]
                                ,[self.frame8],[self.frame9] , [self.frame10] , [self.frame11] , [self.frame12],
                                [self.frame13] ,[self.frame14] , [self.frame15] , [self.frame16] ]
            i = 0 # permet de savoir à quel itération du frame , on est c'est pour stocker 
            for item in self.liste_frame : 
                self.load_and_display_image(self.choice_path_image() , item[0] , self.resize , i)
                i+=1

        #Bouton correspondant à la fonction Reset
        button_reset = Button(self.master,text= "reset" , command = self.fonction_reset_photo , width=10, height=5)
        button_reset.place(x=500,y=100)

        #Correspond à la définition de la  box pour la variabilité 
        vcmd = (self.master.register(self.validate))

        s = Spinbox(self.master , from_= 0 , to = 100 , validate = "all" , validatecommand=(vcmd,'%P'))
        s.place(x = 500, y = 220)
        label_spinbox = Label(self.master, text = "variabilité")
        label_spinbox.place(x = 500 , y = 200)

        #Choix du gif à faire apparaitre dans la barre loading
        gif_path_joke = "C:/Users/lukyl/OneDrive/Images/giphy.gif"
        gif_path = "C:/Users/lukyl/OneDrive/Images/giphy_1.gif"
        gif = Image.open(gif_path_joke)

        #découpage du gif , frame par frame 
        self.frames = [ImageTk.PhotoImage(frame) for frame in ImageSequence.Iterator(gif)]
        self.index = 0 #numéro de l'image du gif qu'on sélectionne

        #bouton qui va enclencher la fenêtre loading
        button_chargement = Button(self.master,text="OK",command=self.lancement_des_fonctions_apres_confirmer , width = 15 , height = 3)
        button_chargement.place(x = 125 , y = 560)

        #Le code ici correspond à la création du bouton rouge 
        self.canvas_red = tk.Canvas(self.master, width=100, height=100, bg="black", highlightthickness=0) #on créer un canvas
        self.canvas_red.place(x = 500 , y = 450)
        rayon = 40 #diamètre du cercle rouge/vert
        x_centre, y_centre = 50, 50 #centre du cercle
        self.cercle_rouge = self.canvas_red.create_oval(x_centre - rayon, y_centre - rayon, x_centre + rayon, y_centre + rayon, fill="red", outline="black") #permet de créer un cercle 
        self.canvas_red.bind("<Button-1>" , lambda event : self.canvas_red_click()) #action qui se produit lorsqu'on clique sur le cercle

        #idem pour la création du bouton rouge sauf que c'est avec le bouton vert 
        self.canvas_vert = tk.Canvas(self.master, width=100, height=100, bg="black", highlightthickness=0)
        self.canvas_vert.place(x = 600 , y = 450)
        self.cercle_vert = self.canvas_vert.create_oval(x_centre - rayon, y_centre - rayon, x_centre + rayon, y_centre + rayon, fill="lightgray", outline="black")
        self.canvas_vert.bind("<Button-1>" , lambda event : self.canvas_vert_click())

        #Frame du meurtrier
        self.photo_meurtrier = Frame(self.master,width = 178, height = 218,background="black")
        self.photo_meurtrier.place(x=950,y=250)

        #Bouton pour confirmer le meurtrier
        bouton_confirmer = Button(self.master,text="confirmer" , width= 15 , height = 3 , command = self.final_choice)
        bouton_confirmer.place(x = 322 , y = 560)

        #Bouton pour quitter le logiciel

        bouton_finish = Button(self.master , text = "finish" , width=15 , height = 3 , command = self.finish)
        bouton_finish.place(x=980,y=480)

        #les images complémentaires qu'on a 

        frame_1_complementaire = Frame(self.master , width = 100+HLT , height = 120+HLT)
        frame_1_complementaire.place( x = 620 , y = 175)
        
        frame_2_complementaire = Frame(self.master, width = 100+HLT , height = 120+HLT)
        frame_2_complementaire.place ( x=620 , y = 40 )

        frame_complementaire_path = "C:/Users/lukyl/Music/javascript/__pycache__/graphique/img_align_celeba/000001.jpg"
        frame_complementaire_size = [100,120]

        self.load_image_from_autoencoder(frame_complementaire_path,frame_1_complementaire,frame_complementaire_size)
        self.load_image_from_autoencoder(frame_complementaire_path,frame_2_complementaire,frame_complementaire_size)



###############################################################FONCCCTIONNNNN


    def choice_path_image(self) : # fonction pour choisir une photo aléatoire dans le dossier
        nombre_alea = str(random.randint(1,202599)) #choisi un nombre entre 1 et 200 000 
        while len(nombre_alea) < 6 : 
            nombre_alea = "0" + nombre_alea # si on a choisi 478 , il faut 3 zéros devant 478 pour le path de l'image 000478
        random_image_path = "C:/Users/lukyl/Music/javascript/__pycache__/graphique/img_align_celeba/" + nombre_alea + ".jpg"
        return random_image_path


    def load_and_display_image(self,image_path,frame,resize,i) : #fonction pour charger les images dans un Frame pré-définit
        pil_image = Image.open(image_path) # on charge l'image
        if len(resize) != 0 : #on check si la liste resize est vite , si non , on resize avec x  et y 
            pil_image = pil_image.resize((resize[0],resize[1]),Image.LANCZOS)
        if len(self.liste_frame[i]) > 1 : # si on a 2 élements dans notre liste , alors une image a été précèdemment chargé , on change juste le image_path à la position 1 
            self.liste_frame[i][1] = image_path
        if len(self.liste_frame[i]) == 1 :  # aucune image a été chargé dans le frame
            self.liste_frame[i].append(image_path) # on ajoute image path à la position 1
        tk_image = ImageTk.PhotoImage(pil_image)
     
        #on met le label d'une image dans le frame 
        label = Label(frame,image = tk_image , bd = 0 , highlightcolor= "black" , highlightthickness=4 , highlightbackground="black")
        label.image = tk_image # il faut conserver l'image pour qu'elle soit afficher 
        label.place(x=0,y=0)
        label.bind("<Button-1>",lambda event , f=frame : self.select_frame(f,label)) #lorsqu'on clique sur le frame , on a cette action
    

    def select_frame(self,frame,label) : # fonction pour la sélection de Frame (couleur Rouge)
        if self.red == True :  #dépend si on est en mode algo génétique ou choix final 
    
            i = 0
            #réinitialise le fond de l'ancien frame sélectionné 
            for num_label in self.frame_select : # on check si le frame n'a pas été sélectionné avant , si c'est le cas on le remet en noir 
                if num_label == label :  
                    label.config(highlightbackground = "black" , highlightcolor = "black" )
                    del self.frame_select[i] # on le dégage de la liste des frame sélectionné en rouge
                    print("delete")
                    return
                i+=1
            #met le fond du Frame en rouge si le frame n'a pas été sélectionné avant
            label.config(highlightbackground = "red" , highlightcolor = "red")
            self.frame_select.append(label) 
            
        else :   #correspond au choix final , c'est à dire couleur vert
          
            if len(self.frame_confirm) != 0 : #image déja sélectionné ou pas ???
                if self.frame_confirm[0] == label : 
                    label.config(highlightbackground = "black" , highlightcolor = "black")
                    del self.frame_confirm[0]
                    
            else :  # si jamais sélectionner alors on l'a met en vert 
                label.config(highlightbackground = "green" , highlightcolor = "green")
                self.frame_confirm.append(label)
        
    def fonction_reset_photo(self) : #Fonction permet de changer les photos qui ont été précèdemment sélectionnées.
        i = 0
        for item in self.liste_frame : 
            self.load_and_display_image(self.choice_path_image() , item[0] , self.resize , i)
            i+=1
        
        self.frame_select = [] # on reset tout car on change d'image , rappel ici c'est les frames sélectionnés en rouge
        self.frame_confirm = [] # ici les frames sélectionnés en vert




    def validate(self,P) : #Fonction pour éviter qu'autre chose soit écrit (entre 0 et 100) dans la case écriture 
        try : 
            value = int(P)
            if value >= 0 and value <= 100 : 
                return True
            else : 
                return False
        except ValueError : 
            return False if P else True
        

    def lancement_des_fonctions_apres_confirmer(self) : 
        self.changement_image_genetic_autoencoder()
       
        self.simulate_chargement()
        
    def changement_image_genetic_autoencoder(self) :
        print(" ici " , self.model)
        if len(self.liste_frame[0]) > 1 : 
        
            for item in self.liste_frame : 
                item.pop(1)
    
        liste_path_test = []
        for i in range(16) : 
            liste_path_test.append(self.choice_path_image())
        for i in range(self.number_image) : 
            self.load_image_from_autoencoder(liste_path_test[i],self.liste_frame[i][0],self.resize)

        self.frame_select = []
        self.frame_confirm = []


    def load_image_from_autoencoder(self,image_path,frame,resize) :
        pil_image = Image.open(image_path) # on charge l'image
        if len(resize) != 0 : #on check si la liste resize est vite , si non , on resize avec x  et y 
            pil_image = pil_image.resize((resize[0],resize[1]),Image.LANCZOS)

        tk_image = ImageTk.PhotoImage(pil_image)
     
        #on met le label d'une image dans le frame 
        label = Label(frame,image = tk_image , bd = 0 , highlightcolor= "black" , highlightthickness=4 , highlightbackground="black")
        label.image = tk_image # il faut conserver l'image pour qu'elle soit afficher 
        label.place(x=0,y=0)
        label.bind("<Button-1>",lambda event , f=frame : self.select_frame(f,label)) #lorsqu'on clique sur le frame , on a cette action
    


        
    



    def simulate_chargement(self) : #Fonction permettant de faire la fenêtre de chargement 
        
        self.master.withdraw() #on fait disparaitre l'ancienne fenetre
        
        nouvelle_fenetre = Toplevel(self.master) #nouvelle fenetre (celle du loading)
        nouvelle_fenetre.title("chargement")

        def update_animation(frame) :  # fonction pour changer de frame avec un pas de temps
            image = self.frames[self.index]
            label.configure(image=image)
            self.index = (self.index+1) % len(self.frames)
            self.master.after(100,update_animation,frame)

        label = Label(nouvelle_fenetre , image = self.frames[self.index]) # Image du Frame du gif sélectionné
        label.pack()

        update_animation(self.frames[self.index])

        #Ici est le code de la barre de chargement , il sera retiré. La fin de la fenêtre loading sera un signal du back

        barre_chargement = ttk.Progressbar(nouvelle_fenetre,orient="horizontal",length=300,mode= "determinate")
        barre_chargement.pack(pady = 10) 

        def update_progress(i) : 
            barre_chargement["value"] =  i
            nouvelle_fenetre.update_idletasks()

        def simulate_progress(i) : 
            if i <= 100 :
                update_progress(i)
                nouvelle_fenetre.after(10,lambda : simulate_progress(i+1))
            else : 
                nouvelle_fenetre.destroy()
                self.master.deiconify()
        simulate_progress(0)



    def canvas_red_click(self) : # fonction sur le click du bouton rouge
        if self.canvas_red.itemcget(self.cercle_rouge,"fill") == "lightgray" : # si le bouton rouge était en gris

            self.canvas_red.itemconfig(self.cercle_rouge,fill="red") #on colore le bouton en rouge
            self.canvas_vert.itemconfig(self.cercle_vert,fill="lightgray") # le bouton vert devient gris
            self.red = True # le bool permettant de savoir si on est en mode vert ou rouge devient true
            if len(self.frame_confirm) != 0 : # permet d'enlever le contour vert des photos précèdemment sélectionné
                self.frame_confirm[0].config(highlightbackground = "black" , highlightcolor = "black" ) # si y'a une photo en vert , on le retire
                del self.frame_confirm[0]
            

    def canvas_vert_click(self) : # idem qu'avec le bouton rouge sauf pour le bouton vert 
        # même principe qu'en haut
        if self.canvas_vert.itemcget(self.cercle_vert,"fill") == "lightgray" : 
            self.canvas_vert.itemconfig(self.cercle_vert,fill="green")
            self.canvas_red.itemconfig(self.cercle_rouge,fill="lightgray")

            i = 0
            #réinitialise le fond des frames précèdemment sélectionné en rouge
            for num_label in self.frame_select : 
                    num_label.config(highlightbackground = "black" , highlightcolor = "black" )
            self.frame_select = []
            self.red = False

    def final_choice(self) : #permet de copier l'image sélectionner par un vert pour l'afficher sur le Frame meurtrier
        """Manipulation bizarre ici , en gros il est difficile de retrouver l'image sous forme PIL à partir du Label
        j'ai stocké les path des label des liste_frame , puis j'arrive à retrouver le nom du frame issue du label 
        ainsi avec ce numéro de frame , j'arrive à retrouver l'index pour le path du frame correspondant """

        print(str(self.frame_confirm[0]))
        if len(self.frame_confirm) != 0 : 
            if len(str(self.frame_confirm[0])) == 15 : # on récupérer le nom du frame , dans ce frame on connait son numéro
                #Les frames dans tkinter sont numérotés selon leurs ordres de créations 
                #on a 4 frames crée avant donc je mets - 4 dans le nom des frames
                index_frame = int(str(self.frame_confirm[0])[7]) - 4 #La partie du string récupéré est le numéro du Frame selon le nom donné par Tkinter
                #exemple si on a Frame15 , alors on récupère le 15 puis on soustrait par 8 car 8 Frame ont été crées avant
            else : 
                index_frame = int(str(self.frame_confirm[0])[7:9]) - 4 # si on a Frame 15 , on doit récupérer le 1 et le 5 c'est pourquoi [7:9]
                #7 est la position initial du numéro dans le string 
            print("ici index" , index_frame)

            """C'est une bidouille très sensible , il faudra changer la valeur parfois car en rajoutant des frames 
            ça va biaiser notre valeur 4 , donc faudra ajuster à chaque fois qu'on fait des modifications sur l'interface"""
            

            #On peut enfin resize l'image avec notre manipulation compliqué !!
            image_meurtrier = Image.open(self.liste_frame[index_frame][1]) 
            image_meurtrier = image_meurtrier.resize((178,218),Image.LANCZOS)
            tk_image_meurtrier = ImageTk.PhotoImage(image_meurtrier)
            

            label_meurtrier = Label(self.photo_meurtrier,image=tk_image_meurtrier)
            label_meurtrier.image = tk_image_meurtrier
            label_meurtrier.place(x=0,y=0)
            self.bool_finish = True # Le tueur a été confirmer on peut avant quitter l'appli
        
    def finish(self) : #permet de quitter la fenetre ( Fin )
        if self.bool_finish == True : #permet d'éviter qu'on quitte l'appli tant qu'on a pas trouver le tueur 
             self.master.quit()

logiciel = Logicielprincipal(Tk())
logiciel.mainloop()