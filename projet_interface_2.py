from tkinter import * 
from tkinter import messagebox
from tkinter import PhotoImage
from PIL import Image, ImageTk , ImageSequence
import random
import time
from tkinter import ttk
import tkinter as tk


fenetre = Tk()

frame_select = []
frame_confirm = []
red = True
bool_finish = False



fenetre.withdraw()
nouvelle_fenetre_menu = Toplevel(fenetre)
nouvelle_fenetre_menu.title("Menu")
nouvelle_fenetre_menu.geometry("600x400")






def menu_sortie() : 
     nouvelle_fenetre_menu.destroy()
     fenetre.deiconify()


nouvelle_fenetre_button = Button(nouvelle_fenetre_menu , text = "Logiciel principal" , command = menu_sortie)
nouvelle_fenetre_button.pack()


class MenuDeroulant(tk.Frame) : 
    def __init__(self,master=None) : 
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self) : 

        #création de la zone de contenu caché 
        self.menu_contenu = Frame(self.master,bg = "lightgrey" , width = 200)
        self.menu_contenu.pack_propagate(False)
        self.menu_contenu.place(x=-150,y=0,relheight=1,relwidth=0.2)

        #création de la zone bouton pour affciher le menu 

        self.bouton_menu = Button(self.master,text="Menu",command=self.toggle_menu)
        self.bouton_menu.place(x=10,y=10)

        self.label_menu = Label(self.menu_contenu,text="Menu items" , bg= "lightgrey")
        self.label_menu.pack(pady=10)
        
        self.items1 = Button(self.menu_contenu,text= "Option 1")
        self.items1.pack(pady=5)

        self.items2 = Button(self.menu_contenu,text = "Option 2")
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
        print("test")
        if self.menu_contenu.winfo_x() < 0 : 
            self.move_contenu()    
        else : 
            self.menu_contenu.place(x=-150)

menu_deroulant = MenuDeroulant(master=nouvelle_fenetre_menu)


"""
class Ajout_image(tk.Frame) : 
    def __init__(self,master=None) :
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets_bag()

    def create_widgets_bag(self) :
        ###############################

"""


        




background_image_1 = Image.open("C:/Users/lukyl/OneDrive/Images/vector_background.jpg")
background_image_1 = background_image_1.resize((1200,800) , Image.LANCZOS)
background_image_2 = ImageTk.PhotoImage(background_image_1)



background_label = Label(fenetre,image = background_image_2 ,)
background_label.place(x=0,y=0,relwidth=1,relheight=1)

fenetre.geometry(f"{background_image_1.width}x{background_image_1.height}")

frame_support_centre  = Frame(fenetre,width=400,height = 550 , background="light blue")
frame_support_centre.place(x=82, y = 80)

frame_fond_meurtrier = Frame(fenetre,width= 200 , height = 330 , background = "light blue")
frame_fond_meurtrier.place(x=936 , y = 230)

frame_centre_ligne = Frame(fenetre,width=750,height = 100 , background = "light blue")
frame_centre_ligne.place(x=350 , y = 300)


fleche_image = Image.open("C:/Users/lukyl/OneDrive/Images/fleche_bleu_t.png")
fleche_image_size = fleche_image.resize((200,95) , Image.LANCZOS)
fleche_image_tk = ImageTk.PhotoImage(fleche_image_size)

test_label = Label(fenetre,image = fleche_image_tk , background="light blue")
test_label.place(x = 500 , y = 300)
test_label_copy = Label(fenetre,image = fleche_image_tk , background = "light blue")
test_label_copy.place(x=700, y = 300)



#nb max 202599
def choice_path_image() : 
    nombre_alea = str(random.randint(1,202599))
    while len(nombre_alea) < 6 : 
        nombre_alea = "0" + nombre_alea
    random_image_path = "C:/Users/lukyl/Music/javascript/__pycache__/graphique/img_align_celeba/" + nombre_alea + ".jpg"
    return random_image_path

#fonction pour charger les images 
def load_and_display_image(image_path,frame) : 
    pil_image = Image.open(image_path)
    tk_image = ImageTk.PhotoImage(pil_image)
    label = Label(frame,image = tk_image , bd = 0 , highlightcolor= "black" , highlightthickness=4 , highlightbackground="black")
    label.image = tk_image
    #label.pack(fill = BOTH , expand =TRUE)
    label.place(x=0,y=0)
    #ajout d'evenement sur click d'une image 
    label.bind("<Button-1>",lambda event , f=frame : select_frame(f,label))

def select_frame(frame,label) : 
    if red == True : 
        global frame_select
        i = 0
        #réinitialise le fond de l'ancien frame sélectionné 
        for num_label in frame_select : 
            if num_label == label :  
                label.config(highlightbackground = "black" , highlightcolor = "black" )
                del frame_select[i]
                print("delete")
                return
            i+=1
        
        label.config(highlightbackground = "red" , highlightcolor = "red")
        frame_select.append(label) 
        print("done")
    else : 
        print("test")
        global frame_confirm
        print(len(frame_confirm))
        if len(frame_confirm) != 0 : 
            if frame_confirm[0] == label : 
                label.config(highlightbackground = "black" , highlightcolor = "black")
                del frame_confirm[0]
                
        else :  
            label.config(highlightbackground = "green" , highlightcolor = "green")
            frame_confirm.append(label)
    
        

#créer quatre autres frames 
HLT = 8
frame1 = Frame(fenetre, bd=0, width=178 + HLT, height=218+HLT)  # Spécifiez la taille du frame
frame2 = Frame(fenetre, bd=0, width=178+HLT, height=218+HLT)
frame3 = Frame(fenetre, bd=0, width=178+HLT, height=218+HLT)
frame4 = Frame(fenetre, bd=0, width=178+HLT, height=218+HLT)


frame1.place(x=100-HLT, y=100-HLT)  # Spécifiez les coordonnées en pixels
frame2.place(x=100+178+HLT, y=100-HLT)
frame3.place(x=100-HLT, y=100+218+HLT)
frame4.place(x=100+178+HLT, y=100+218+HLT)


load_and_display_image(choice_path_image() , frame1)
load_and_display_image(choice_path_image() , frame2)
load_and_display_image(choice_path_image() , frame3)
load_and_display_image(choice_path_image() , frame4)


def fonction_reset_photo() : 
    global frame_select
    global frame_confirm
    
    load_and_display_image(choice_path_image() , frame1)
    load_and_display_image(choice_path_image() , frame2)
    load_and_display_image(choice_path_image() , frame3)
    load_and_display_image(choice_path_image() , frame4)
    
    frame_select = []
    frame_confirm = []

button_reset = Button(fenetre,text= "reset" , command = fonction_reset_photo , width=10, height=5)
button_reset.place(x=500,y=100)


def validate(P) : 
    
    
    try : 
        value = int(P)
        if value >= 0 and value <= 100 : 
            return True
        else : 
            return False
    except ValueError : 
        return False if P else True
    

vcmd = (fenetre.register(validate))

s = Spinbox(fenetre , from_= 0 , to = 100 , validate = "all" , validatecommand=(vcmd,'%P'))
s.place(x = 500, y = 220)
label_spinbox = Label(fenetre, text = "variabilité")
label_spinbox.place(x = 500 , y = 200)

gif_path_joke = "C:/Users/lukyl/OneDrive/Images/giphy.gif"
gif_path = "C:/Users/lukyl/OneDrive/Images/giphy_1.gif"
gif = Image.open(gif_path)

frames = [ImageTk.PhotoImage(frame) for frame in ImageSequence.Iterator(gif)]

index = 0



def simulate_chargement() : 
    global frames
    global index
    #on fait disparaitre l'ancienne fenetre
    fenetre.withdraw()
    #nouvelle fenetre
    nouvelle_fenetre = Toplevel(fenetre)
    nouvelle_fenetre.title("chargement")

    def update_animation(frame) : 
        global index
        image = frames[index]
        label.configure(image=image)
        index = (index+1) % len(frames)
        fenetre.after(100,update_animation,frame)

    label = Label(nouvelle_fenetre , image = frames[index])
    label.pack()

    update_animation(frames[index])



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
            fenetre.deiconify()
    simulate_progress(0)

button_chargement = Button(fenetre,text="OK",command=simulate_chargement , width = 15 , height = 3)
button_chargement.place(x = 125 , y = 560)


def canvas_red_click() : 
    if canvas_red.itemcget(cercle_rouge,"fill") == "lightgray" : 
        global red
        global frame_confirm
        canvas_red.itemconfig(cercle_rouge,fill="red")
        canvas_vert.itemconfig(cercle_vert,fill="lightgray")
        red = True
        if len(frame_confirm) != 0 : 
             frame_confirm[0].config(highlightbackground = "black" , highlightcolor = "black" )
             del frame_confirm[0]
        

canvas_red = tk.Canvas(fenetre, width=100, height=100, bg="black", highlightthickness=0)
canvas_red.place(x = 500 , y = 450)
rayon = 40
x_centre, y_centre = 50, 50
cercle_rouge = canvas_red.create_oval(x_centre - rayon, y_centre - rayon, x_centre + rayon, y_centre + rayon, fill="red", outline="black")
canvas_red.bind("<Button-1>" , lambda event : canvas_red_click())


def canvas_vert_click() : 
    if canvas_vert.itemcget(cercle_vert,"fill") == "lightgray" : 
        canvas_vert.itemconfig(cercle_vert,fill="green")
        canvas_red.itemconfig(cercle_rouge,fill="lightgray")

        global frame_select
        global red 
        i = 0
        #réinitialise le fond des frames précèdeent sélectionné 
        print(len(frame_select))
        for num_label in frame_select : 
                
                print("got  one")
                num_label.config(highlightbackground = "black" , highlightcolor = "black" )
        frame_select = []
        red = False

canvas_vert = tk.Canvas(fenetre, width=100, height=100, bg="black", highlightthickness=0)
canvas_vert.place(x = 600 , y = 450)
cercle_vert = canvas_vert.create_oval(x_centre - rayon, y_centre - rayon, x_centre + rayon, y_centre + rayon, fill="lightgray", outline="black")
canvas_vert.bind("<Button-1>" , lambda event : canvas_vert_click())

def final_choice() : 
    global frame_confirm
    global bool_finish
    if len(frame_confirm) != 0 : 
        image_meurtrier = frame_confirm[0].cget("image")
        label_meurtrier = Label(photo_meurtrier,image=image_meurtrier)
        label_meurtrier.place(x=0,y=0)
        bool_finish = True
    



photo_meurtrier = Frame(fenetre,width = 178, height = 218,background="black")
photo_meurtrier.place(x=950,y=250)

bouton_confirmer = Button(fenetre,text="confirmer" , width= 15 , height = 3 , command = final_choice)
bouton_confirmer.place(x = 322 , y = 560)

def finish() : 
    fenetre.quit()

bouton_finish = Button(fenetre , text = "finish" , width=15 , height = 3 , command = finish)
bouton_finish.place(x=980,y=480)

fenetre.mainloop()