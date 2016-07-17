from Tkinter import *
from StockPredictor.dataset_construction import populate_webdata


class Window(Tk):
	
	def __init__(self, *args, **kwargs):
		Tk.__init__(self, *args, **kwargs)
		
		Tk.wm_title(self, "Stock Return Predictor")
		
		container = Frame(self)
		container.pack(side="top", fill="both", expand=True)
		container.grid_rowconfigure(0, weight=1)
		container.grid_columnconfigure(0, weight=1)
		
		menubar = Menu(container)
		filemenu = Menu(menubar, tearoff=0)
		filemenu.add_command(label="Repopulate", command=populate_webdata)
		filemenu.add_separator()
		filemenu.add_command(label="Exit", command=quit)
		menubar.add_cascade(label="File", menu=filemenu)
		
		Tk.config(self, menu=menubar)
		
		self.frames = {}
		
		for F in (Page, Page):
			frame = F(container, self)
			
			self.frames[F] = frame
			
			frame.grid(row=0, column=0, sticky="nsew")
		
		self.show_frame(Page)
		
	def show_frame(self, cont):
		
		frame = self.frames[cont]
		frame.tkraise()

class Page(Frame):
	
	def __init__(self, parent, controller):
		Frame.__init__(self, parent)
		label = Label(self, text="Currently Not in Use")

application = Window()
application.geometry("300x200")
application.mainloop()