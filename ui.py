import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import numpy as np

class UserInterface:

    def __init__(self, root, population):
        self.root = root
        self.root.title("Attacker identifier")
        self.root.geometry("1000x600")
        self.root.resizable(False, False)

        self.main_label = tk.Label(self.root, text="Click on the image that most resembles your attacker.")
        self.main_label.grid(row=0)

        self.population = population

        self.image1 = ImageTk.PhotoImage(Image.fromarray(self.population[0]))
        self.button1 = tk.Button(self.root, text='1', image=self.image1)
        self.button1.configure(command=lambda: self.on_image_click(self.button1))
        self.button1.grid(row=1, column=0)

        self.image2 = ImageTk.PhotoImage(Image.fromarray(self.population[1]))
        self.button2 = tk.Button(self.root, text='2', image=self.image2)
        self.button2.configure(command=lambda: self.on_image_click(self.button2))
        self.button2.grid(row=1, column=1)

        self.image3 = ImageTk.PhotoImage(Image.fromarray(self.population[2]))
        self.button3 = tk.Button(self.root, text='3', image=self.image3)
        self.button3.configure(command=lambda: self.on_image_click(self.button3))
        self.button3.grid(row=2, column=0)

        self.image4 = ImageTk.PhotoImage(Image.fromarray(self.population[3]))
        self.button4 = tk.Button(self.root, text='4', image=self.image4)
        self.button4.configure(command=lambda: self.on_image_click(self.button4))
        self.button4.grid(row=2, column=1)

        self.validationButton = tk.Button(self.root, text='Validate', command=self.on_validate_click)
        self.validationButton.grid(sticky='se')
        
        self.user_choice = []
        self.choices_validated = False   #identify_attacker.py waits for this to be True before continuing

    def on_image_click(self, button):
        image_number = int(button.cget('text'))
        image = button.cget('image')
        if image not in self.user_choice:
            self.user_choice.append(image)
        self.buttonClickedLabel = tk.Label(self.root, text=f"You clicked on image {image_number}")
        self.buttonClickedLabel.grid(row=5, column=0)
        #return image_number

    def on_validate_click(self):
        self.choice_validated = True
        # choice = []
        # for image_number in self.which_image_clicked:
        #     choice.append(image_number)
        # return choice
    
    def display_new_images(self, new_population):
        self.population = new_population
        self.user_choice = []
        self.choices_validated = False

        self.image1 = self.population[0]
        self.button1 = tk.Button(self.root, text='1', image=self.image1, command=self.on_image_click(self.button1))
        self.button1.grid(row=1, column=0)

        self.image2 = self.population[1]
        self.button2 = tk.Button(self.root, text='2', image=self.image2, command=self.on_image_click(self.button2))
        self.button2.grid(row=1, column=1)

        self.image3 = self.population[2]
        self.button3 = tk.Button(self.root, text='3', image=self.image3, command=self.on_image_click(self.button3))
        self.button3.grid(row=2, column=0)

        self.image4 = self.population[3]
        self.button4 = tk.Button(self.root, text='4', image=self.image4, command=self.on_image_click(self.button4))
        self.button4.grid(row=2, column=1)

def main():
    root = tk.Tk()

    # Generate some random image data
    image_shape = (100, 100)  # Image dimensions
    num_images = 4

    # Create a list to store the image arrays
    dummy_population = []

    # Generate random images and append them to the list
    for _ in range(num_images):
        # Generate random values for the image array
        image_array = np.random.randint(0, 256, size=image_shape, dtype=np.uint8)
        dummy_population.append(image_array)
    
    population = dummy_population
    app = UserInterface(root, population)
    root.mainloop()

if __name__ == "__main__":
    main()

# The UserInterface class is a simple
# class that initializes a window and
# adds a label and a button to it.
# The main() function creates an instance
# of the UserInterface class and runs the
# application.