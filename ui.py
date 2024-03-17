import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
# import sys

class UserInterface:

    def __init__(self, root, population):
        self.root = root
        self.root.title("Attacker identifier")
        self.root.geometry("1000x600")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.root.quit)  # Not working

        self.population = population
        self.iteration = 1
        self.user_choice = []
        self.choices_validated = False   #identify_attacker.py waits for this to be True before continuing

        self.main_label = tk.Label(self.root, text="Click on the image that most resembles your attacker.")
        self.main_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, ipadx=10, ipady=10)

        self.image1 = ImageTk.PhotoImage(Image.fromarray((self.population[0] * 255).astype(np.uint8)))
        self.button1 = tk.Button(self.root, text='1', image=self.image1)
        self.button1.configure(command=lambda: self.on_image_click(self.button1))
        self.button1.grid(row=1, column=0, padx=10, pady=10)

        self.image2 = ImageTk.PhotoImage(Image.fromarray((self.population[1] * 255).astype(np.uint8)))
        self.button2 = tk.Button(self.root, text='2', image=self.image2)
        self.button2.configure(command=lambda: self.on_image_click(self.button2))
        self.button2.grid(row=1, column=1, padx=10, pady=10)

        self.image3 = ImageTk.PhotoImage(Image.fromarray((self.population[2] * 255).astype(np.uint8)))
        self.button3 = tk.Button(self.root, text='3', image=self.image3)
        self.button3.configure(command=lambda: self.on_image_click(self.button3))
        self.button3.grid(row=2, column=0, padx=10, pady=10)

        self.image4 = ImageTk.PhotoImage(Image.fromarray((self.population[3] * 255).astype(np.uint8)))
        self.button4 = tk.Button(self.root, text='4', image=self.image4)
        self.button4.configure(command=lambda: self.on_image_click(self.button4))
        self.button4.grid(row=2, column=1, padx=10, pady=10)

        self.buttonClickedLabel = tk.Label(self.root, text="You have clicked on the following images: ")
        self.buttonClickedLabel.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        self.validationButton = tk.Button(self.root, text='Validate', command=self.on_validate_click)
        self.validationButton.grid(row=4, column=0, columnspan=2, padx=10, pady=10, ipadx=75, ipady=5)

        self.iteration_label = tk.Label(self.root, text=f'Iteration: {self.iteration}')
        self.iteration_label.grid(row=5, column=0, columnspan=2, padx=10, pady=10)

        # self.fin0_label = tk.Label(self.root, text="fin")
        # self.fin0_label.grid(row=6, column=0, sticky = 'e')

        # self.debut0_label = tk.Label(self.root, text="debut")
        # self.debut0_label.grid(row=6, column=0, sticky = 'w')

        # self.fin1_label = tk.Label(self.root, text="fin")
        # self.fin1_label.grid(row=6, column=1, sticky = 'e')

        # self.debut1_label = tk.Label(self.root, text="debut")
        # self.debut1_label.grid(row=6, column=1, sticky = 'w')


    # # Define a function to stop the program
    # def stop_program(self):
    #     self.root.quit()     # stops mainloop
    #     self.root.destroy()  # this is necessary on Windows to prevent
    #                     # Fatal Python Error: PyEval_RestoreThread: NULL tstate
    #     sys.exit()      # stops the rest of the script

    def on_image_click(self, button):
        image_number = int(button.cget('text'))
        if image_number not in self.user_choice:
            self.user_choice.append(image_number)
        self.buttonClickedLabel.config(text=f"You have clicked on the following images: {self.user_choice}")

    def on_validate_click(self):
        self.choices_validated = True
        # print("Validation button clicked")
    
    def display_new_images(self, new_population):
        self.population = new_population
        self.user_choice = []
        self.choices_validated = False
        self.buttonClickedLabel.config(text="You have clicked on the following images: ")

        self.iteration += 1
        self.iteration_label.config(text=f'Iteration: {self.iteration}')
        
        self.image1 = ImageTk.PhotoImage(Image.fromarray((self.population[0] * 255).astype(np.uint8)))
        self.button1.config(image=self.image1, command=lambda: self.on_image_click(self.button1))

        self.image2 = ImageTk.PhotoImage(Image.fromarray((self.population[1] * 255).astype(np.uint8)))
        self.button2.config(image=self.image2, command=lambda: self.on_image_click(self.button2))

        self.image3 = ImageTk.PhotoImage(Image.fromarray((self.population[2] * 255).astype(np.uint8)))
        self.button3.config(image=self.image3, command=lambda: self.on_image_click(self.button3))

        self.image4 = ImageTk.PhotoImage(Image.fromarray((self.population[3] * 255).astype(np.uint8)))
        self.button4.config(image=self.image4, command=lambda: self.on_image_click(self.button4))

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