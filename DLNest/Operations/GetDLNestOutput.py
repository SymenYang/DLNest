from DLNest.Output.DLNestBuffer import DLNestBuffer

def getDLNestOutput(
    style : bool = True
):
    buffer = DLNestBuffer()

    if style:
        return buffer.getStyledText()
    else:
        return buffer.getPlainText()