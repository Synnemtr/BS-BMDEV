"""
Module: user_interface.py

This module provides a simple user interface for identifying attackers using images.

The UserInterface class initializes a window and adds a label and buttons for selecting images.
"""

import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import sys

class UserInterface:
    """
    A class to represent the user interface for identifying attackers using images.

    Attributes:
        root (Tk): The root window of the user interface.
        population (list): A list containing image arrays representing the population of images.
        iteration (int): The current iteration of image selection.
        user_choice (list): A list of integers representing the indices of images selected by the user.
        choices_validated (bool): A boolean indicating whether the user's choices have been validated.
        more_iterations (bool): A boolean indicating whether there are more iterations for image selection.
    """
    def __init__(self, root, population):
        """
        Initializes the user interface with a root window and adds necessary widgets.

        Args:
            root (Tk): The root window of the user interface.
            population (list): A list containing image arrays representing the population of images.
        """
        self.root = root
        self.root.title("Attacker identifier")
        self.root.geometry("340x535")
        self.root.resizable(False, False)
        self.root.protocol("WM_DELETE_WINDOW", self.stop_program)

        self.window_exists = True
        self.population = population
        self.iteration = 1
        self.user_choice = []
        self.choices_validated = False   #identify_attacker.py waits for this to be True before continuing
        self.more_iterations = False

        # Create a label to display the main message
        self.main_label = tk.Label(self.root, text="Click on the image(s) that most resemble your attacker.")
        self.main_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, ipadx=10, ipady=10)

        # Create the clickable images on the tkinter window
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

        # Create a label to display the images that the user has clicked on
        self.buttonClickedLabel = tk.Label(self.root, text="You have clicked on the following images: ")
        self.buttonClickedLabel.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

        # Create a button to validate the user's choice
        self.validationButton = tk.Button(self.root, text='Validate', command=self.on_validate_click)
        self.validationButton.grid(row=4, column=0, columnspan=2, padx=10, pady=10, ipadx=75, ipady=5)

        # Create a label to display the current iteration
        self.iteration_label = tk.Label(self.root, text=f'Iteration: {self.iteration}')
        self.iteration_label.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    # Define a function to stop the program when the window is closed
    def stop_program(self):
        """
        Stop the program when the window is closed.

        This method quits the mainloop of the Tkinter root window and destroys the window.
        It also updates the status of the window existence flag and prints a message indicating the window closure.
        """
        self.root.quit()                # stops mainloop
        self.root.destroy()             # this is necessary on Windows to prevent
                                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate
        self.window_exists = False
        print("Window closed")

    # Define a function to determine the actions performed when the user clicks on an image
    def on_image_click(self, button):
        """
        Perform actions when the user clicks on an image.

        Args:
            button (Button): The button widget representing the clicked image.

        This method retrieves the number associated with the clicked image button and
        updates the user's choice list accordingly. It also updates the label to display
        the images that the user has clicked on.
        """
        image_number = int(button.cget('text'))
        if image_number not in self.user_choice:
            self.user_choice.append(image_number)
        self.buttonClickedLabel.config(text=f"You have clicked on the following images: {self.user_choice}")

    # Define a function to determine the actions performed when the user clicks on the "Validate" button
    def on_validate_click(self):
        """
        Perform actions when the user clicks on the "Validate" button.

        This method checks if the user has selected at least one image.
        If not, it displays an error message. Otherwise, it sets the choices_validated flag to True.
        """
        if len(self.user_choice) == 0:
            self.error_label = tk.Label(self.root, text="You must select at least one image.", foreground="red")
            self.error_label.grid(row=5, column=0, columnspan=2)
        else :
            if hasattr(self, 'error_label'):
                self.error_label.destroy()
            self.choices_validated = True
    
    # Define a function to display the new images after the user has validated their choice
    def display_new_images(self, new_population):
        """
        Display new images after the user has validated their choice.

        Args:
            new_population (list): A list containing new image arrays representing the population of images.

        This method updates the population of images with the new population.
        It resets the user's choice list, sets the choices_validated flag to False,
        and updates the buttonClickedLabel to display an empty list of clicked images.
        Additionally, it updates the iteration count and the labels and commands of image buttons
        to reflect the changes in the population.
        """
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

    # Define a function to continue the program after the user has reached the maximum number of iterations
    # def continue_program(self, population_if_continue):
    #     # à modifier : end_window à la place de end_screen pour ne pas tout supprimer
    #     # sur la fenêtre principale et pouvoir continuer à sélectionner des images

    #     self.population = population_if_continue
    #     self.iteration = 1
    #     self.user_choice = []
    #     self.choices_validated = False   #identify_attacker.py waits for this to be True before continuing
    #     self.more_iterations = True

    #     # Change the label displaying the main message
    #     self.main_label.config(text="Click on the image(s) that most resemble your attacker.")

    #     # Deleting the final picture and the buttons
    #     self.final_picture.destroy()
    #     self.closing_button.destroy()
    #     self.continue_button.destroy()

    #     # Create the clickable images on the tkinter window
    #     self.image1 = ImageTk.PhotoImage(Image.fromarray((self.population[0] * 255).astype(np.uint8)))
    #     self.button1 = tk.Button(self.root, text='1', image=self.image1)
    #     self.button1.configure(command=lambda: self.on_image_click(self.button1))
    #     self.button1.grid(row=1, column=0, padx=10, pady=10)

    #     self.image2 = ImageTk.PhotoImage(Image.fromarray((self.population[1] * 255).astype(np.uint8)))
    #     self.button2 = tk.Button(self.root, text='2', image=self.image2)
    #     self.button2.configure(command=lambda: self.on_image_click(self.button2))
    #     self.button2.grid(row=1, column=1, padx=10, pady=10)

    #     self.image3 = ImageTk.PhotoImage(Image.fromarray((self.population[2] * 255).astype(np.uint8)))
    #     self.button3 = tk.Button(self.root, text='3', image=self.image3)
    #     self.button3.configure(command=lambda: self.on_image_click(self.button3))
    #     self.button3.grid(row=2, column=0, padx=10, pady=10)

    #     self.image4 = ImageTk.PhotoImage(Image.fromarray((self.population[3] * 255).astype(np.uint8)))
    #     self.button4 = tk.Button(self.root, text='4', image=self.image4)
    #     self.button4.configure(command=lambda: self.on_image_click(self.button4))
    #     self.button4.grid(row=2, column=1, padx=10, pady=10)

    #     # Create a label to display the images that the user has clicked on
    #     self.buttonClickedLabel = tk.Label(self.root, text="You have clicked on the following images: ")
    #     self.buttonClickedLabel.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

    #     # Create a button to validate the user's choice
    #     self.validationButton = tk.Button(self.root, text='Validate', command=self.on_validate_click)
    #     self.validationButton.grid(row=4, column=0, columnspan=2, padx=10, pady=10, ipadx=75, ipady=5)

    #     # Create a label to display the current iteration
    #     self.iteration_label = tk.Label(self.root, text=f'Iteration: {self.iteration}')
    #     self.iteration_label.grid(row=6, column=0, columnspan=2, padx=10, pady=10)

    #     self.root.mainloop()

    def end_screen(self, population):
        """
        Display the end screen with the final image of the attacker.

        Args:
            population (list): A list containing the final image array of the attacker.

        This method prepares and displays the end screen of the user interface.
        It sets the final individual's image, resets various attributes, and destroys
        the widgets related to image selection. It then configures the main label to
        inform the user about reaching the maximum number of iterations and displays
        the final image of the attacker. Additionally, it creates a button to close
        the application.

        Note:
            This method assumes the root window is already initialized and running.
        """
        self.final_individual = population
        # self.population_if_continue = population_if_continue
        self.user_choice = []
        self.choices_validated = False
        self.more_iterations = False
        # print('Variables changed')

        self.buttonClickedLabel.destroy()
        self.iteration_label.destroy()
        self.button1.destroy()
        self.button2.destroy()
        self.button3.destroy()
        self.button4.destroy()
        self.validationButton.destroy()
        # print('Buttons destroyed')

        self.main_label.config(text="You reached the maximum number of iterations. Final image of the attacker :")

        self.image1 = ImageTk.PhotoImage(Image.fromarray((self.final_individual[0] * 255).astype(np.uint8)))
        self.final_picture = tk.Label(self.root, image=self.image1)
        self.final_picture.grid(row=1, column=0, columnspan=2, padx=10, pady=10)

        self.closing_button = tk.Button(self.root, text="Close", command=self.stop_program)
        self.closing_button.grid(row=2, column=0, columnspan=2, padx=10, pady=10, ipadx=20, ipady=10)
        # self.continue_button = tk.Button(self.root, text="Continue selecting images", command=lambda: self.continue_program(self.population_if_continue))
        # self.continue_button.grid(row=2, column=1, padx=10, pady=10, ipadx=20, ipady=10)

        self.root.mainloop()

def main():
    """
    Main function to initialize the user interface and run the application.

    This function serves as the entry point of the application. It creates the root
    window using Tkinter, generates random image data, initializes the user interface
    with the generated population of images, and starts the main event loop to handle
    user interactions.

    The main purpose of this function is to start the graphical user interface (GUI)
    application and ensure its smooth execution by running the main event loop.
    """
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