from OutputLayer import *
from prompt_toolkit import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.layout.containers import VSplit, Window,HSplit
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.layout import Layout
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit import print_formatted_text, HTML
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import FormattedText,to_formatted_text
import os
import sys
import time


buffer1 = Buffer()  # Editable buffer.

w2 = Window(content=FormattedTextControl(text='Hello world'))

root_container = VSplit([
    # One window that holds the BufferControl with the default buffer on
    # the left.
    w2
])


kb = KeyBindings()

@kb.add('c-q')
def exit_(event):
    """
    Pressing Ctrl-Q will exit the user interface.

    Setting a return value means: quit the event loop that drives the user
    interface and return this value from the `Application.run()` call.
    """
    event.app.exit()


@kb.add('c-m')
def add_message(event):
    ol = AnalyzerBuffer()
    ol.logMessage("test message")
    now_offset,styled_text = ol.getStyledText()
    text = FormattedText(styled_text)
    w2.content.text = text

@kb.add('c-i')
def add_ign(event):
    ol = AnalyzerBuffer()
    ol.logIgnError("test ign error")
    now_offset,styled_text = ol.getStyledText()
    text = FormattedText(styled_text)
    w2.content.text = text


@kb.add('c-t')
def test_error(event):
    ol = AnalyzerBuffer()
    old = sys.stdout
    sys.stdout = ol
    print(len(ol.styled_text))
    sys.stdout = old

@kb.add('c-y')
def test_error(event):
    ol = AnalyzerBuffer()
    now_offset,styled_text = ol.getStyledText()
    text = FormattedText(styled_text)
    w2.content.text = text

@kb.add('c-e')
def add_error(event):
    ol = AnalyzerBuffer()
    ol.logError("test error")
    now_offset,styled_text = ol.getStyledText()
    text = FormattedText(styled_text)
    w2.content.text = text


@kb.add('c-d')
def add_debug(event):
    ol = AnalyzerBuffer()
    ol.logDebug("test debug message")
    now_offset,styled_text = ol.getStyledText()
    text = FormattedText(styled_text)
    w2.content.text = text

layout = Layout(root_container)
app = Application(key_bindings=kb, layout=layout, full_screen=True)
app.run() # You won't be able to Exit this app
#with open("test.txt",'w') as f:
#    tr = TrainStdout(f)
#    old = sys.stdout
#    sys.stdout = f
#    for i in range(10):
#        print(i)
#        time.sleep(3)
#    sys.stdout = old
