Attribute VB_Name = "modCustodian"
' ================================================================================
' modCustodian -- Custodian PID extraction & LTV enrichment
' Standalone module, no dependency on modConfig/modMain/modSteps
' ================================================================================
Option Explicit

' --- Sheet layout ---
Private Const CUST_SHEET As String = "Sheet1"
Private Const CUST_START_ROW As Long = 2
Private Const CUST_NAME_COL As Long = 1 ' Col A
Private Const CUST_COLLATERAL_COL As Long = 5 ' Col E
Private Const CUST_COMMITTED_COL As Long = 6 ' Col F
Private Const CUST_PID_UNIQUE_COL As Long = 7 ' Col G
Private Const CUST_PID_LIST_COL As Long = 8 ' Col H
Private Const CUST_CRDS_COL As Long = 9 ' Col I
Private Const CUST_PID_TOTAL_COL As Long = 10 ' Col J

' --- Zone affichage taux FX ---
Private Const FX_DISPLAY_START_ROW As Long = 25
Private Const FX_LABEL_COL As Long = 1 ' Col A
Private Const FX_VALUE_COL As Long = 2 ' Col B

' --- Network path ---
Private Const NET_FOLDER As String = "\\dfs\root\Fo\Appli\hftbpss\eod\"
Private Const RAW_PREFIX As String = "RAWRISK"
Private Const RAW_SUFFIX As String = "_OTHERS"
Private Const LTV_PREFIX As String = "LTVNOTNYK"
Private Const LTV_SUFFIX As String = ""

' --- Chemin fichier FX ---
Private Const FX_LOG_FOLDER As String = "\\dfs\root\Fo\Appli\hftbpss\log\MDD\"
Private Const FX_LOG_PREFIX As String = "Fx_"
Private Const FX_LOG_SUFFIX As String = ".log"

' --- RAWRISK columns (0-indexed) ---
Private Const RAW_COL_PID As Long = 0 ' Col A
Private Const RAW_COL_CUSTODIAN As Long = 63 ' Col BL

' --- LTVNOTNYK columns (0-indexed) ---
Private Const LTV_COL_PID As Long = 0 ' Col A
Private Const LTV_COL_CURRENCY As Long = 3 ' Col D
Private Const LTV_COL_COMMITTED As Long = 4 ' Col E
Private Const LTV_COL_COLLATERAL As Long = 8 ' Col I
Private Const LTV_COL_CRDS As Long = 54 ' Col BC

' --- Date business (module-level, calculee une seule fois) ---
Private m_businessDate As String


' ================================================================================
' LOGGING
' ================================================================================
Private Sub CustLogInit()
    Dim wsLog As Worksheet
    On Error Resume Next
    Set wsLog = ActiveWorkbook.Sheets("Log")
    On Error GoTo 0
    If wsLog Is Nothing Then
        Set wsLog = ActiveWorkbook.Sheets.Add( _
            After:=ActiveWorkbook.Sheets(ActiveWorkbook.Sheets.Count))
        wsLog.Name = "Log"
    End If
    wsLog.Cells.Interior.Color = RGB(255, 255, 255)
    wsLog.Range("A1").Value = "Timestamp"
    wsLog.Range("B1").Value = "Status"
    wsLog.Range("C1").Value = "Message"
    wsLog.Range("A1:C1").Font.Bold = True
    wsLog.Range("A1:C1").Interior.Color = RGB(44, 62, 80)
    wsLog.Range("A1:C1").Font.Color = RGB(255, 255, 255)
    wsLog.Range("A1:C1").Font.Size = 10
    wsLog.Range("A1:C1").HorizontalAlignment = xlCenter
    wsLog.Rows(1).RowHeight = 24
    wsLog.Columns("A").ColumnWidth = 20
    wsLog.Columns("B").ColumnWidth = 10
    wsLog.Columns("C").ColumnWidth = 100
    wsLog.Columns("C").NumberFormat = "@"
End Sub

Private Sub CustLog(ByVal msg As String, _
                    Optional ByVal status As String = "INFO")
    Dim wsLog As Worksheet
    Dim nr As Long
    On Error Resume Next
    Set wsLog = ActiveWorkbook.Sheets("Log")
    On Error GoTo 0
    If wsLog Is Nothing Then Exit Sub
    On Error GoTo LogErr
    nr = wsLog.Cells(wsLog.Rows.Count, 1).End(xlUp).Row + 1
    wsLog.Range("A" & nr & ":C" & nr).NumberFormat = "@"
    wsLog.Cells(nr, 1).Value = Format(Now, "yyyy-mm-dd hh:mm:ss")
    wsLog.Cells(nr, 2).Value = status
    wsLog.Cells(nr, 3).Value = msg
    wsLog.Range("A" & nr & ":C" & nr).Font.Size = 10
    wsLog.Cells(nr, 1).Font.Color = RGB(127, 140, 141)
    wsLog.Cells(nr, 2).Font.Bold = True
    wsLog.Cells(nr, 2).HorizontalAlignment = xlCenter
    Select Case UCase(status)
        Case "ERROR"
            wsLog.Range("A" & nr & ":C" & nr).Interior.Color = RGB(250, 219, 216)
            wsLog.Cells(nr, 2).Font.Color = RGB(192, 57, 43)
        Case "WARN"
            wsLog.Range("A" & nr & ":C" & nr).Interior.Color = RGB(254, 249, 231)
            wsLog.Cells(nr, 2).Font.Color = RGB(183, 149, 11)
        Case "OK"
            wsLog.Range("A" & nr & ":C" & nr).Interior.Color = RGB(234, 250, 234)
            wsLog.Cells(nr, 2).Font.Color = RGB(39, 174, 96)
        Case "INFO"
            If nr Mod 2 = 0 Then
                wsLog.Range("A" & nr & ":C" & nr).Interior.Color = RGB(245, 247, 250)
            Else
                wsLog.Range("A" & nr & ":C" & nr).Interior.Color = RGB(255, 255, 255)
            End If
            wsLog.Cells(nr, 2).Font.Color = RGB(52, 73, 94)
    End Select
    wsLog.Range("A" & nr & ":C" & nr).Borders(xlEdgeBottom).LineStyle = xlContinuous
    wsLog.Range("A" & nr & ":C" & nr).Borders(xlEdgeBottom).Color = RGB(230, 230, 230)
    wsLog.Range("A" & nr & ":C" & nr).Borders(xlEdgeBottom).Weight = xlThin
    On Error GoTo 0
    Debug.Print Format(Now, "hh:mm:ss") & " [" & status & "] " & msg
    Exit Sub
LogErr:
    On Error GoTo 0
    Debug.Print "CustLog FAILED: " & msg
End Sub

Private Sub CustLogClear()
    Dim wsLog As Worksheet
    Dim lr As Long
    On Error Resume Next
    Set wsLog = ActiveWorkbook.Sheets("Log")
    On Error GoTo 0
    If Not wsLog Is Nothing Then
        lr = wsLog.Cells(wsLog.Rows.Count, 1).End(xlUp).Row
        If lr > 1 Then wsLog.Range("A2:C" & lr).Clear
    End If
End Sub


' ================================================================================
' HELPERS
' ================================================================================

Private Function FindLastBusinessDate() As String
    Dim d As Date
    Dim i As Long
    Dim testPath As String

    If Len(m_businessDate) > 0 Then
        FindLastBusinessDate = m_businessDate
        Exit Function
    End If

    d = Date - 1
    For i = 1 To 10
        testPath = NET_FOLDER & RAW_PREFIX & _
                   Format(d, "YYYYMMDD") & RAW_SUFFIX
        If FileExists(testPath) Then
            CustLog "FindLastBusinessDate: " & Format(d, "YYYYMMDD") & _
                    " trouve (J-" & i & ")", "OK"
            m_businessDate = Format(d, "YYYYMMDD")
            FindLastBusinessDate = m_businessDate
            Exit Function
        End If
        CustLog "FindLastBusinessDate: " & Format(d, "YYYYMMDD") & _
                " absent, recul...", "INFO"
        d = d - 1
    Next i
    CustLog "FindLastBusinessDate: aucun fichier sur 10 jours!", "ERROR"
    m_businessDate = Format(Date - 1, "YYYYMMDD")
    FindLastBusinessDate = m_businessDate
End Function

Private Function FindLastLTVDate() As String
    Dim d As Date
    Dim i As Long
    Dim testPath As String

    d = Date - 1
    For i = 1 To 10
        testPath = NET_FOLDER & LTV_PREFIX & _
                   Format(d, "YYYYMMDD") & LTV_SUFFIX
        If FileExists(testPath) Then
            CustLog "FindLastLTVDate: " & Format(d, "YYYYMMDD") & _
                    " trouve (J-" & i & ")", "OK"
            FindLastLTVDate = Format(d, "YYYYMMDD")
            Exit Function
        End If
        CustLog "FindLastLTVDate: " & Format(d, "YYYYMMDD") & _
                " absent, recul...", "INFO"
        d = d - 1
    Next i
    CustLog "FindLastLTVDate: aucun fichier sur 10 jours!", "ERROR"
    FindLastLTVDate = Format(Date - 1, "YYYYMMDD")
End Function

Private Function FindLastFxDate() As String
    Dim d As Date
    Dim i As Long
    Dim testPath As String

    d = Date - 1
    For i = 1 To 10
        testPath = FX_LOG_FOLDER & FX_LOG_PREFIX & _
                   Format(d, "YYYYMMDD") & FX_LOG_SUFFIX
        If FileExists(testPath) Then
            CustLog "FindLastFxDate: " & Format(d, "YYYYMMDD") & _
                    " trouve (J-" & i & ")", "OK"
            FindLastFxDate = Format(d, "YYYYMMDD")
            Exit Function
        End If
        CustLog "FindLastFxDate: " & Format(d, "YYYYMMDD") & _
                " absent, recul...", "INFO"
        d = d - 1
    Next i
    CustLog "FindLastFxDate: aucun fichier FX sur 10 jours!", "ERROR"
    FindLastFxDate = ""
End Function

Private Function BuildRawPath() As String
    BuildRawPath = NET_FOLDER & RAW_PREFIX & _
                   FindLastBusinessDate() & RAW_SUFFIX
End Function

Private Function BuildLTVPath() As String
    BuildLTVPath = NET_FOLDER & LTV_PREFIX & _
                   FindLastLTVDate() & LTV_SUFFIX
End Function

Private Function FetchEURRates(ByRef fxRates As Object) As Boolean
    Dim fxDate As String
    Dim filePath As String
    Dim lines() As String
    Dim i As Long
    Dim oneLine As String
    Dim spotPos As Long
    Dim eqPos As Long
    Dim pairPart As String
    Dim valuePart As String
    Dim rawRate As Double
    Dim eurUsd As Double
    Dim gbpUsd As Double
    Dim jpyUsd As Double
    Dim chfUsd As Double
    Dim cadUsd As Double

    Set fxRates = CreateObject("Scripting.Dictionary")
    fxRates("EUR") = 1
    eurUsd = 0
    gbpUsd = 0
    jpyUsd = 0
    chfUsd = 0
    cadUsd = 0

    fxDate = FindLastFxDate()
    If Len(fxDate) = 0 Then
        CustLog "FetchEURRates: aucun fichier FX disponible", "ERROR"
        FetchEURRates = False
        Exit Function
    End If

    filePath = FX_LOG_FOLDER & FX_LOG_PREFIX & fxDate & FX_LOG_SUFFIX
    CustLog "FetchEURRates: Fichier: " & filePath, "INFO"

    If Not SafeReadFile(filePath, lines) Then
        FetchEURRates = False
        Exit Function
    End If

    For i = 0 To UBound(lines)
        oneLine = Trim(lines(i))
        If Len(oneLine) = 0 Then GoTo NextFxLine

        spotPos = InStr(oneLine, ".spot")
        If spotPos = 0 Then GoTo NextFxLine

        eqPos = InStr(spotPos, oneLine, "=")
        If eqPos = 0 Then GoTo NextFxLine

        pairPart = UCase(Trim(Left(oneLine, spotPos - 1)))
        valuePart = Trim(Mid(oneLine, eqPos + 1))
        valuePart = Replace(valuePart, ",", ".")
        rawRate = Val(valuePart)

        If rawRate <= 0 Then GoTo NextFxLine

        Select Case pairPart
            Case "EUR_USD"
                eurUsd = rawRate
                CustLog "FetchEURRates: EUR_USD.spot = " & _
                    Format(rawRate, "0.000000"), "INFO"
            Case "GBP_USD"
                gbpUsd = rawRate
                CustLog "FetchEURRates: GBP_USD.spot = " & _
                    Format(rawRate, "0.000000"), "INFO"
            Case "JPY_USD"
                jpyUsd = rawRate
                CustLog "FetchEURRates: JPY_USD.spot = " & _
                    Format(rawRate, "0.000000"), "INFO"
            Case "CHF_USD"
                chfUsd = rawRate
                CustLog "FetchEURRates: CHF_USD.spot = " & _
                    Format(rawRate, "0.000000"), "INFO"
            Case "CAD_USD"
                cadUsd = rawRate
                CustLog "FetchEURRates: CAD_USD.spot = " & _
                    Format(rawRate, "0.000000"), "INFO"
        End Select
NextFxLine:
    Next i

    If eurUsd = 0 Then
        CustLog "FetchEURRates: EUR_USD introuvable!", "ERROR"
        FetchEURRates = False
        Exit Function
    End If

    fxRates("USD") = eurUsd
    CustLog "FetchEURRates: EUR/USD = " & Format(eurUsd, "0.0000"), "OK"

    If gbpUsd > 0 Then
        fxRates("GBP") = eurUsd / gbpUsd
        CustLog "FetchEURRates: EUR/GBP = " & _
            Format(eurUsd / gbpUsd, "0.0000") & _
            " (= " & Format(eurUsd, "0.0000") & _
            " / " & Format(gbpUsd, "0.0000") & ")", "OK"
    Else
        CustLog "FetchEURRates: GBP_USD introuvable!", "WARN"
    End If

    If jpyUsd > 0 Then
        fxRates("JPY") = eurUsd / jpyUsd
        CustLog "FetchEURRates: EUR/JPY = " & _
            Format(eurUsd / jpyUsd, "0.0000") & _
            " (= " & Format(eurUsd, "0.0000") & _
            " / " & Format(jpyUsd, "0.0000") & ")", "OK"
    Else
        CustLog "FetchEURRates: JPY_USD introuvable!", "WARN"
    End If

    If chfUsd > 0 Then
        fxRates("CHF") = eurUsd / chfUsd
        CustLog "FetchEURRates: EUR/CHF = " & _
            Format(eurUsd / chfUsd, "0.0000") & _
            " (= " & Format(eurUsd, "0.0000") & _
            " / " & Format(chfUsd, "0.0000") & ")", "OK"
    Else
        CustLog "FetchEURRates: CHF_USD introuvable", "WARN"
    End If

    If cadUsd > 0 Then
        fxRates("CAD") = eurUsd / cadUsd
        CustLog "FetchEURRates: EUR/CAD = " & _
            Format(eurUsd / cadUsd, "0.0000") & _
            " (= " & Format(eurUsd, "0.0000") & _
            " / " & Format(cadUsd, "0.0000") & ")", "OK"
    Else
        CustLog "FetchEURRates: CAD_USD introuvable", "WARN"
    End If

    CustLog "FetchEURRates: " & fxRates.Count & " devises", "OK"
    FetchEURRates = True
End Function

Private Sub WriteFxDisplay(fxRates As Object)
    Dim ws As Worksheet
    Dim r As Long

    On Error Resume Next
    Set ws = ActiveWorkbook.Sheets(CUST_SHEET)
    On Error GoTo 0
    If ws Is Nothing Then Exit Sub

    r = FX_DISPLAY_START_ROW

    If fxRates.Exists("USD") Then
        ws.Cells(r, FX_LABEL_COL).Value = "Rate USD/EUR"
        ws.Cells(r, FX_VALUE_COL).Value = 1 / CDbl(fxRates("USD"))
        ws.Cells(r, FX_VALUE_COL).NumberFormat = "0.000000"
        ws.Cells(r, FX_LABEL_COL).Font.Bold = True
        r = r + 1
    End If

    If fxRates.Exists("GBP") Then
        ws.Cells(r, FX_LABEL_COL).Value = "Rate GBP/EUR"
        ws.Cells(r, FX_VALUE_COL).Value = 1 / CDbl(fxRates("GBP"))
        ws.Cells(r, FX_VALUE_COL).NumberFormat = "0.000000"
        ws.Cells(r, FX_LABEL_COL).Font.Bold = True
        r = r + 1
    End If

    If fxRates.Exists("JPY") Then
        ws.Cells(r, FX_LABEL_COL).Value = "Rate JPY/EUR"
        ws.Cells(r, FX_VALUE_COL).Value = 1 / CDbl(fxRates("JPY"))
        ws.Cells(r, FX_VALUE_COL).NumberFormat = "0.000000"
        ws.Cells(r, FX_LABEL_COL).Font.Bold = True
        r = r + 1
    End If

    If fxRates.Exists("CHF") Then
        ws.Cells(r, FX_LABEL_COL).Value = "Rate CHF/EUR"
        ws.Cells(r, FX_VALUE_COL).Value = 1 / CDbl(fxRates("CHF"))
        ws.Cells(r, FX_VALUE_COL).NumberFormat = "0.000000"
        ws.Cells(r, FX_LABEL_COL).Font.Bold = True
        r = r + 1
    End If

    If fxRates.Exists("CAD") Then
        ws.Cells(r, FX_LABEL_COL).Value = "Rate CAD/EUR"
        ws.Cells(r, FX_VALUE_COL).Value = 1 / CDbl(fxRates("CAD"))
        ws.Cells(r, FX_VALUE_COL).NumberFormat = "0.000000"
        ws.Cells(r, FX_LABEL_COL).Font.Bold = True
        r = r + 1
    End If

    CustLog "WriteFxDisplay: taux ecrits lignes " & _
        FX_DISPLAY_START_ROW & "-" & (r - 1), "OK"
End Sub

Private Function ToEUR(ByVal amount As Double, _
                       ByVal ccy As String, _
                       fxRates As Object) As Double
    ccy = UCase(Trim(ccy))
    If ccy = "EUR" Or Len(ccy) = 0 Then
        ToEUR = amount
        Exit Function
    End If
    If fxRates.Exists(ccy) Then
        If CDbl(fxRates(ccy)) <> 0 Then
            ToEUR = amount / CDbl(fxRates(ccy))
        Else
            ToEUR = amount
        End If
    Else
        CustLog "ToEUR: devise inconnue """ & ccy & _
            """, montant non converti", "WARN"
        ToEUR = amount
    End If
End Function

Private Function FileExists(ByVal filePath As String) As Boolean
    On Error GoTo NotFound
    FileExists = (Dir(filePath) <> "")
    Exit Function
NotFound:
    FileExists = False
End Function

Private Function DetectSeparator(ByVal firstLine As String) As String
    Dim tabCount As Long
    Dim semiCount As Long
    Dim commaCount As Long
    Dim i As Long
    For i = 1 To Len(firstLine)
        Select Case Mid(firstLine, i, 1)
            Case vbTab: tabCount = tabCount + 1
            Case ";": semiCount = semiCount + 1
            Case ",": commaCount = commaCount + 1
        End Select
    Next i
    If tabCount >= semiCount And tabCount >= commaCount And tabCount > 0 Then
        DetectSeparator = vbTab
        CustLog "Separateur: TAB (" & tabCount & ")", "OK"
    ElseIf semiCount >= commaCount And semiCount > 0 Then
        DetectSeparator = ";"
        CustLog "Separateur: POINT-VIRGULE (" & semiCount & ")", "WARN"
    ElseIf commaCount > 0 Then
        DetectSeparator = ","
        CustLog "Separateur: VIRGULE (" & commaCount & ")", "WARN"
    Else
        DetectSeparator = vbTab
        CustLog "Aucun separateur, fallback TAB", "WARN"
    End If
End Function

Private Function SafeReadFile(ByVal filePath As String, _
                              ByRef lines() As String) As Boolean
    Dim fileNum As Integer
    Dim fileContent As String
    Dim fileOpened As Boolean
    fileOpened = False
    On Error GoTo ReadError
    fileNum = FreeFile
    Open filePath For Input As #fileNum
    fileOpened = True
    If LOF(fileNum) = 0 Then
        Close #fileNum
        CustLog "Fichier vide: " & filePath, "ERROR"
        SafeReadFile = False
        Exit Function
    End If
    fileContent = Input$(LOF(fileNum), fileNum)
    Close #fileNum
    fileOpened = False
    If Len(fileContent) >= 3 Then
        If Asc(Mid(fileContent, 1, 1)) = 239 And _
           Asc(Mid(fileContent, 2, 1)) = 187 And _
           Asc(Mid(fileContent, 3, 1)) = 191 Then
            fileContent = Mid(fileContent, 4)
            CustLog "BOM UTF-8 supprime", "INFO"
        End If
    End If
    lines = Split(fileContent, vbCrLf)
    CustLog "Fichier lu: " & filePath & " (" & _
        (UBound(lines) + 1) & " lignes)", "OK"
    SafeReadFile = True
    Exit Function
ReadError:
    If fileOpened Then Close #fileNum
    CustLog "ERREUR lecture: " & filePath & " -- " & _
        Err.Description, "ERROR"
    SafeReadFile = False
End Function

Private Function ConvertUS(ByVal usValue As String) As Double
    usValue = Trim(usValue)
    If Len(usValue) = 0 Then
        ConvertUS = 0
        Exit Function
    End If
    usValue = Replace(usValue, ",", "")
    ConvertUS = Val(usValue)
End Function

Private Function HasEnoughCols(fields() As String, _
                               required As Long, context As String) As Boolean
    If UBound(fields) < required Then
        CustLog context & ": attendu " & (required + 1) & _
            " colonnes, trouve " & (UBound(fields) + 1), "WARN"
        HasEnoughCols = False
    Else
        HasEnoughCols = True
    End If
End Function


' ================================================================================
' ETAPE 1
' ================================================================================
Public Function EtapeCust1_ReadCustodians( _
        ByRef custNames As Object) As Boolean
    Dim ws As Worksheet
    Dim r As Long
    Dim v As String
    Dim nameList As String
    Dim parts() As String
    Dim p As Long
    Dim oneName As String

    Set custNames = CreateObject("Scripting.Dictionary")

    On Error Resume Next
    Set ws = ActiveWorkbook.Sheets(CUST_SHEET)
    On Error GoTo 0
    If ws Is Nothing Then
        CustLog "EtapeCust1: Feuille """ & CUST_SHEET & """ introuvable", "ERROR"
        EtapeCust1_ReadCustodians = False
        Exit Function
    End If

    If IsEmpty(ws.Cells(CUST_START_ROW, CUST_NAME_COL).Value) Then
        CustLog "EtapeCust1: Cellule A" & CUST_START_ROW & " vide", "ERROR"
        EtapeCust1_ReadCustodians = False
        Exit Function
    End If

    r = CUST_START_ROW
    Do
        On Error Resume Next
        v = ""
        v = Trim(CStr(ws.Cells(r, CUST_NAME_COL).Value))
        On Error GoTo 0
        If Len(v) = 0 Then Exit Do
        parts = Split(v, "/")
        For p = 0 To UBound(parts)
            oneName = UCase(Trim(parts(p)))
            If Len(oneName) > 0 Then
                If custNames.Exists(oneName) Then
                    CustLog "EtapeCust1: Doublon ignore: " & oneName & " (ligne " & r & ")", "WARN"
                Else
                    custNames.Add oneName, r
                    nameList = nameList & Trim(parts(p)) & ", "
                End If
            End If
        Next p
        r = r + 1
    Loop

    If custNames.Count = 0 Then
        CustLog "EtapeCust1: Aucun custodian", "ERROR"
        EtapeCust1_ReadCustodians = False
        Exit Function
    End If

    If Len(nameList) > 2 Then nameList = Left(nameList, Len(nameList) - 2)
    CustLog "EtapeCust1: " & custNames.Count & " noms (alias inclus): " & nameList, "OK"
    EtapeCust1_ReadCustodians = True
End Function


' ================================================================================
' ETAPE 2
' ================================================================================
Public Function EtapeCust2_ReadRawRisk( _
        custNames As Object, _
        ByRef custPIDs As Object) As Boolean
    Dim filePath As String
    Dim lines() As String
    Dim fields() As String
    Dim sep As String
    Dim i As Long
    Dim custKey As String
    Dim pid As String
    Dim matchCount As Long
    Dim skipHeader As Boolean
    Dim k As Variant
    Dim firstDataLine As Long
    Dim innerDict As Object
    Dim cnt As Long
    Dim pk As Variant
    Dim pidDebug As String
    Dim custList As String

    Set custPIDs = CreateObject("Scripting.Dictionary")
    For Each k In custNames.Keys
        Set innerDict = CreateObject("Scripting.Dictionary")
        Set custPIDs(k) = innerDict
    Next k

    filePath = BuildRawPath()
    CustLog "EtapeCust2: Fichier: " & filePath, "INFO"

    If Not FileExists(filePath) Then
        CustLog "EtapeCust2: RAWRISK introuvable", "ERROR"
        EtapeCust2_ReadRawRisk = False
        Exit Function
    End If
    If Not SafeReadFile(filePath, lines) Then
        EtapeCust2_ReadRawRisk = False
        Exit Function
    End If

    sep = DetectSeparator(lines(0))
    fields = Split(lines(0), sep)
    CustLog "EtapeCust2: " & (UBound(fields) + 1) & " colonnes", "INFO"

    If UBound(fields) >= RAW_COL_PID Then
        CustLog "EtapeCust2: Col " & RAW_COL_PID & " header=[" & Trim(fields(RAW_COL_PID)) & "]", "INFO"
    End If
    If UBound(fields) >= RAW_COL_CUSTODIAN Then
        CustLog "EtapeCust2: Col " & RAW_COL_CUSTODIAN & " header=[" & Trim(fields(RAW_COL_CUSTODIAN)) & "]", "INFO"
    End If

    skipHeader = False
    If UBound(fields) >= RAW_COL_PID Then
        If Not IsNumeric(Trim(fields(RAW_COL_PID))) Then
            skipHeader = True
            CustLog "EtapeCust2: Header detecte", "INFO"
        End If
    End If

    firstDataLine = IIf(skipHeader, 1, 0)
    If firstDataLine > UBound(lines) Then
        CustLog "EtapeCust2: Aucune donnee", "ERROR"
        EtapeCust2_ReadRawRisk = False
        Exit Function
    End If
    fields = Split(lines(firstDataLine), sep)
    If Not HasEnoughCols(fields, RAW_COL_CUSTODIAN, "EtapeCust2") Then
        CustLog "EtapeCust2: Colonnes insuffisantes", "ERROR"
        EtapeCust2_ReadRawRisk = False
        Exit Function
    End If

    CustLog "EtapeCust2: 1ere donnee -> PID=[" & Trim(fields(RAW_COL_PID)) & "] CUST=[" & Trim(fields(RAW_COL_CUSTODIAN)) & "]", "INFO"

    custList = ""
    For Each k In custNames.Keys
        custList = custList & "[" & k & "] "
    Next k
    CustLog "EtapeCust2: Recherche: " & custList, "INFO"

    matchCount = 0
    For i = firstDataLine To UBound(lines)
        If Len(Trim(lines(i))) = 0 Then GoTo NextRawLine
        On Error Resume Next
        fields = Split(lines(i), sep)
        On Error GoTo 0
        If UBound(fields) < RAW_COL_CUSTODIAN Then GoTo NextRawLine
        custKey = UCase(Trim(fields(RAW_COL_CUSTODIAN)))
        If custPIDs.Exists(custKey) Then
            pid = Trim(fields(RAW_COL_PID))
            If Len(pid) > 0 Then
                If Not custPIDs(custKey).Exists(pid) Then
                    custPIDs(custKey).Add pid, 1
                Else
                    custPIDs(custKey)(pid) = custPIDs(custKey)(pid) + 1
                End If
                matchCount = matchCount + 1
            End If
        End If
NextRawLine:
    Next i

    For Each k In custNames.Keys
        cnt = custPIDs(k).Count
        If cnt = 0 Then
            CustLog "EtapeCust2: " & k & " -> 0 PIDs", "WARN"
        Else
            pidDebug = ""
            For Each pk In custPIDs(k).Keys
                pidDebug = pidDebug & CStr(pk) & "(x" & custPIDs(k)(pk) & "), "
            Next pk
            If Len(pidDebug) > 2 Then pidDebug = Left(pidDebug, Len(pidDebug) - 2)
            CustLog "EtapeCust2: " & k & " -> " & cnt & " PIDs: " & pidDebug, "OK"
        End If
    Next k

    CustLog "EtapeCust2: " & (UBound(lines) + 1) & " lignes, " & matchCount & " matches", "OK"
    EtapeCust2_ReadRawRisk = True
End Function


' ================================================================================
' ETAPE 3
' ================================================================================
Public Function EtapeCust3_WriteRawResults( _
        custNames As Object, _
        custPIDs As Object) As Boolean
    Dim ws As Worksheet
    Dim k As Variant
    Dim row As Long
    Dim pidList As String
    Dim pidKey As Variant
    Dim errCount As Long
    Dim readH As String
    Dim rowPIDs As Object
    Dim merged As Object
    Dim rowKey As Variant
    Dim pidUnique As Long
    Dim aliasKey As Variant
    Dim formulaJ As String
    Dim pidAppear As Long

    On Error Resume Next
    Set ws = ActiveWorkbook.Sheets(CUST_SHEET)
    On Error GoTo 0
    If ws Is Nothing Then
        CustLog "EtapeCust3: Feuille introuvable", "ERROR"
        EtapeCust3_WriteRawResults = False
        Exit Function
    End If

    Set rowPIDs = CreateObject("Scripting.Dictionary")
    For Each k In custNames.Keys
        row = custNames(k)
        If Not rowPIDs.Exists(row) Then
            Set merged = CreateObject("Scripting.Dictionary")
            Set rowPIDs(row) = merged
        End If
        If custPIDs.Exists(k) Then
            For Each pidKey In custPIDs(k).Keys
                If Not rowPIDs(row).Exists(CStr(pidKey)) Then
                    rowPIDs(row).Add CStr(pidKey), True
                End If
            Next pidKey
            CustLog "EtapeCust3: Alias [" & k & "] -> ligne " & row & " ajoute " & custPIDs(k).Count & " PIDs (total: " & rowPIDs(row).Count & ")", "INFO"
        End If
    Next k

    errCount = 0
    For Each rowKey In rowPIDs.Keys
        row = CLng(rowKey)
        On Error GoTo WriteFail3

        pidList = ""
        pidUnique = 0
        formulaJ = ""

        For Each pidKey In rowPIDs(rowKey).Keys
            If Len(pidList) > 0 Then pidList = pidList & "/"
            pidList = pidList & CStr(pidKey)
            pidUnique = pidUnique + 1

            pidAppear = 0
            For Each aliasKey In custNames.Keys
                If custNames(aliasKey) = row Then
                    If custPIDs.Exists(aliasKey) Then
                        If custPIDs(aliasKey).Exists(CStr(pidKey)) Then
                            pidAppear = pidAppear + CLng(custPIDs(aliasKey)(CStr(pidKey)))
                        End If
                    End If
                End If
            Next aliasKey

            If Len(formulaJ) > 0 Then formulaJ = formulaJ & "+"
            formulaJ = formulaJ & pidAppear & "+N(""" & CStr(pidKey) & """)"
        Next pidKey

        CustLog "EtapeCust3: Ligne " & row & " -> " & pidUnique & " uniques, formule=[=" & Left(formulaJ, 60) & "]", "INFO"

        ws.Cells(row, CUST_PID_LIST_COL).NumberFormat = "@"
        ws.Cells(row, CUST_PID_LIST_COL).Value = pidList
        ws.Cells(row, CUST_PID_UNIQUE_COL).Value = pidUnique

        If Len(formulaJ) > 0 Then
            ws.Cells(row, CUST_PID_TOTAL_COL).Formula = "=" & formulaJ
        Else
            ws.Cells(row, CUST_PID_TOTAL_COL).Value = 0
        End If

        readH = CStr(ws.Cells(row, CUST_PID_LIST_COL).Value)
        If readH <> pidList Then
            CustLog "EtapeCust3: VERIF ECHOUEE ligne " & row, "ERROR"
            errCount = errCount + 1
        Else
            CustLog "EtapeCust3: Ligne " & row & " OK: H=" & Left(pidList, 50) & " G=" & pidUnique, "OK"
        End If

        GoTo NextCust3
WriteFail3:
        CustLog "EtapeCust3: ERREUR ligne " & row & ": " & Err.Description, "ERROR"
        errCount = errCount + 1
        Err.Clear
        Resume NextCust3
NextCust3:
        On Error GoTo 0
    Next rowKey

    EtapeCust3_WriteRawResults = (errCount = 0)
    If errCount > 0 Then
        CustLog "EtapeCust3: " & errCount & " erreur(s)", "ERROR"
    Else
        CustLog "EtapeCust3: OK", "OK"
    End If
End Function


' ================================================================================
' ETAPE 4
' ================================================================================
Public Function EtapeCust4_ReadLTV( _
        custPIDs As Object, _
        ByRef ltvData As Object) As Boolean
    Dim filePath As String
    Dim lines() As String
    Dim fields() As String
    Dim sep As String
    Dim i As Long
    Dim pid As String
    Dim skipHeader As Boolean
    Dim firstDataLine As Long
    Dim k As Variant
    Dim pk As Variant
    Dim committed As Double
    Dim collateral As Double
    Dim crds As String
    Dim ccy As String
    Dim foundCount As Long
    Dim missingCount As Long
    Dim allPIDs As Object

    Set ltvData = CreateObject("Scripting.Dictionary")
    Set allPIDs = CreateObject("Scripting.Dictionary")
    For Each k In custPIDs.Keys
        For Each pk In custPIDs(k).Keys
            If Not allPIDs.Exists(CStr(pk)) Then allPIDs.Add CStr(pk), True
        Next pk
    Next k

    CustLog "EtapeCust4: " & allPIDs.Count & " PIDs a chercher", "INFO"
    If allPIDs.Count = 0 Then
        CustLog "EtapeCust4: Aucun PID -- SKIP", "WARN"
        EtapeCust4_ReadLTV = True
        Exit Function
    End If

    filePath = BuildLTVPath()
    CustLog "EtapeCust4: Fichier: " & filePath, "INFO"

    If Not FileExists(filePath) Then
        CustLog "EtapeCust4: LTV introuvable", "ERROR"
        EtapeCust4_ReadLTV = False
        Exit Function
    End If
    If Not SafeReadFile(filePath, lines) Then
        EtapeCust4_ReadLTV = False
        Exit Function
    End If

    sep = DetectSeparator(lines(0))
    fields = Split(lines(0), sep)
    skipHeader = False
    If UBound(fields) >= LTV_COL_PID Then
        If Not IsNumeric(Trim(fields(LTV_COL_PID))) Then
            skipHeader = True
            CustLog "EtapeCust4: Header detecte", "INFO"
        End If
    End If

    firstDataLine = IIf(skipHeader, 1, 0)
    If firstDataLine > UBound(lines) Then
        CustLog "EtapeCust4: Aucune donnee", "ERROR"
        EtapeCust4_ReadLTV = False
        Exit Function
    End If
    fields = Split(lines(firstDataLine), sep)
    If Not HasEnoughCols(fields, LTV_COL_CRDS, "EtapeCust4") Then
        CustLog "EtapeCust4: Colonnes insuffisantes", "ERROR"
        EtapeCust4_ReadLTV = False
        Exit Function
    End If

    foundCount = 0
    For i = firstDataLine To UBound(lines)
        If Len(Trim(lines(i))) = 0 Then GoTo NextLTVLine
        On Error Resume Next
        fields = Split(lines(i), sep)
        On Error GoTo 0
        If UBound(fields) < LTV_COL_CRDS Then GoTo NextLTVLine

        pid = Trim(fields(LTV_COL_PID))
        If allPIDs.Exists(pid) Then
            committed = ConvertUS(fields(LTV_COL_COMMITTED))
            collateral = ConvertUS(fields(LTV_COL_COLLATERAL))
            crds = Trim(fields(LTV_COL_CRDS))
            ccy = UCase(Trim(fields(LTV_COL_CURRENCY)))
            ltvData(pid) = Array(committed, collateral, crds, ccy)
            foundCount = foundCount + 1
            CustLog "EtapeCust4: PID " & pid & " (" & ccy & ") committed=" & Format(committed, "#,##0.00") & " collateral=" & Format(collateral, "#,##0.00") & " crds=[" & crds & "]", "INFO"
        End If
NextLTVLine:
    Next i

    missingCount = 0
    For Each pk In allPIDs.Keys
        If Not ltvData.Exists(CStr(pk)) Then
            CustLog "EtapeCust4: PID " & pk & " MANQUANT", "WARN"
            missingCount = missingCount + 1
        End If
    Next pk

    CustLog "EtapeCust4: " & foundCount & "/" & allPIDs.Count & " trouves (" & missingCount & " manquants)", "OK"
    EtapeCust4_ReadLTV = True
End Function


' ================================================================================
' ETAPE 5
' ================================================================================
Public Function EtapeCust5_WriteLTVResults( _
        custNames As Object, _
        custPIDs As Object, _
        ltvData As Object, _
        fxRates As Object) As Boolean
    Dim ws As Worksheet
    Dim k As Variant
    Dim pidKey As Variant
    Dim row As Long
    Dim errCount As Long
    Dim ltvArr As Variant
    Dim crdsVal As String
    Dim pidCcy As String
    Dim rowKey As Variant
    Dim convCommitted As Double
    Dim convCollateral As Double
    Dim rawCommitted As Double
    Dim rawCollateral As Double
    Dim fxRate As Double
    Dim partE As String
    Dim partF As String
    Dim fE As String
    Dim fF As String
    Dim rowFormulaE As Object
    Dim rowFormulaF As Object
    Dim rowCRDS As Object
    Dim rowTotalE As Object
    Dim rowTotalF As Object

    Set rowFormulaE = CreateObject("Scripting.Dictionary")
    Set rowFormulaF = CreateObject("Scripting.Dictionary")
    Set rowCRDS = CreateObject("Scripting.Dictionary")
    Set rowTotalE = CreateObject("Scripting.Dictionary")
    Set rowTotalF = CreateObject("Scripting.Dictionary")

    On Error Resume Next
    Set ws = ActiveWorkbook.Sheets(CUST_SHEET)
    On Error GoTo 0
    If ws Is Nothing Then
        CustLog "EtapeCust5: Feuille introuvable", "ERROR"
        EtapeCust5_WriteLTVResults = False
        Exit Function
    End If

    For Each k In custNames.Keys
        row = custNames(k)
        If Not rowFormulaE.Exists(row) Then
            rowFormulaE(row) = ""
            rowFormulaF(row) = ""
            rowCRDS(row) = ""
            rowTotalE(row) = 0#
            rowTotalF(row) = 0#
        End If
        If custPIDs.Exists(k) Then
            For Each pidKey In custPIDs(k).Keys
                If ltvData.Exists(CStr(pidKey)) Then
                    ltvArr = ltvData(CStr(pidKey))
                    pidCcy = CStr(ltvArr(3))
                    rawCommitted = CDbl(ltvArr(0))
                    rawCollateral = CDbl(ltvArr(1))

                    If UCase(Trim(pidCcy)) = "EUR" Or Len(Trim(pidCcy)) = 0 Then
                        fxRate = 1
                    ElseIf fxRates.Exists(UCase(Trim(pidCcy))) Then
                        fxRate = CDbl(fxRates(UCase(Trim(pidCcy))))
                        If fxRate = 0 Then fxRate = 1
                    Else
                        fxRate = 1
                        CustLog "EtapeCust5: devise """ & pidCcy & """ inconnue, pas de conversion", "WARN"
                    End If

                    If fxRate = 1 Then
                        partE = Replace(Format(rawCollateral, "0.00"), ",", ".")
                        partF = Replace(Format(rawCommitted, "0.00"), ",", ".")
                    Else
                        partE = Replace(Format(rawCollateral, "0.00"), ",", ".") & "/" & Replace(Format(fxRate, "0.0000"), ",", ".")
                        partF = Replace(Format(rawCommitted, "0.00"), ",", ".") & "/" & Replace(Format(fxRate, "0.0000"), ",", ".")
                    End If

                    partE = partE & "+N(""" & CStr(pidKey) & """)"
                    partF = partF & "+N(""" & CStr(pidKey) & """)"

                    If Len(CStr(rowFormulaE(row))) > 0 Then
                        rowFormulaE(row) = CStr(rowFormulaE(row)) & "+" & partE
                    Else
                        rowFormulaE(row) = partE
                    End If
                    If Len(CStr(rowFormulaF(row))) > 0 Then
                        rowFormulaF(row) = CStr(rowFormulaF(row)) & "+" & partF
                    Else
                        rowFormulaF(row) = partF
                    End If

                    convCollateral = rawCollateral / fxRate
                    convCommitted = rawCommitted / fxRate
                    rowTotalE(row) = CDbl(rowTotalE(row)) + convCollateral
                    rowTotalF(row) = CDbl(rowTotalF(row)) + convCommitted

                    CustLog "EtapeCust5: PID " & pidKey & " (" & pidCcy & ") collateral=" & Format(rawCollateral, "#,##0.00") & "/" & Format(fxRate, "0.0000") & " committed=" & Format(rawCommitted, "#,##0.00") & "/" & Format(fxRate, "0.0000"), "INFO"

                    crdsVal = CStr(ltvArr(2))
                    If Len(crdsVal) > 0 Then
                        If Len(CStr(rowCRDS(row))) > 0 Then
                            rowCRDS(row) = CStr(rowCRDS(row)) & "/" & crdsVal
                        Else
                            rowCRDS(row) = crdsVal
                        End If
                    End If
                Else
                    CustLog "EtapeCust5: PID " & pidKey & " absent du LTV", "WARN"
                End If
            Next pidKey
        End If
    Next k

    errCount = 0
    For Each rowKey In rowFormulaE.Keys
        row = CLng(rowKey)
        On Error GoTo WriteFail5

        fE = CStr(rowFormulaE(rowKey))
        fF = CStr(rowFormulaF(rowKey))

        If Len(fE) > 0 Then
            ws.Cells(row, CUST_COLLATERAL_COL).Formula = "=" & fE
        Else
            ws.Cells(row, CUST_COLLATERAL_COL).Value = 0
        End If
        If Len(fF) > 0 Then
            ws.Cells(row, CUST_COMMITTED_COL).Formula = "=" & fF
        Else
            ws.Cells(row, CUST_COMMITTED_COL).Value = 0
        End If

        ws.Cells(row, CUST_CRDS_COL).NumberFormat = "@"
        ws.Cells(row, CUST_CRDS_COL).Value = CStr(rowCRDS(rowKey))

        CustLog "EtapeCust5: Ligne " & row & " formule E=[=" & Left(fE, 60) & "] formule F=[=" & Left(fF, 60) & "] total E=" & Format(CDbl(rowTotalE(rowKey)), "#,##0.00") & " total F=" & Format(CDbl(rowTotalF(rowKey)), "#,##0.00") & " CRDS=" & Left(CStr(rowCRDS(rowKey)), 30), "OK"

        GoTo NextCust5
WriteFail5:
        CustLog "EtapeCust5: ERREUR ligne " & row & ": " & Err.Description, "ERROR"
        errCount = errCount + 1
        Err.Clear
        Resume NextCust5
NextCust5:
        On Error GoTo 0
    Next rowKey

    EtapeCust5_WriteLTVResults = (errCount = 0)
    If errCount > 0 Then
        CustLog "EtapeCust5: " & errCount & " erreur(s)", "ERROR"
    Else
        CustLog "EtapeCust5: OK", "OK"
    End If
End Function


' ================================================================================
' RunCustodian
' ================================================================================
Public Sub RunCustodian()
    Dim startTime As Double
    Dim crashCount As Long
    Dim custNames As Object
    Dim custPIDs As Object
    Dim ltvData As Object
    Dim fxRates As Object
    Dim stepOK As Boolean
    Dim ltvOK As Boolean
    Dim fxOK As Boolean
    Dim elapsed As Double
    Dim ws As Worksheet

    startTime = Timer
    crashCount = 0

    On Error Resume Next
    Set ws = ActiveWorkbook.Sheets(CUST_SHEET)
    On Error GoTo 0
    If ws Is Nothing Then
        MsgBox "Feuille """ & CUST_SHEET & """ introuvable.", vbCritical
        Exit Sub
    End If

    CustLogInit
    CustLogClear
    m_businessDate = ""

    Application.ScreenUpdating = False
    Application.Calculation = xlCalculationManual
    Application.EnableEvents = False

    CustLog "========== RunCustodian ==========", "INFO"
    CustLog "Classeur: " & ActiveWorkbook.Name, "INFO"
    CustLog "Date RAWRISK: " & FindLastBusinessDate(), "INFO"
    CustLog "Date LTV: " & FindLastLTVDate(), "INFO"
    CustLog "RAW: " & BuildRawPath(), "INFO"
    CustLog "LTV: " & BuildLTVPath(), "INFO"

    ' ETAPE 1
    On Error Resume Next
    stepOK = EtapeCust1_ReadCustodians(custNames)
    If Err.Number <> 0 Then
        CustLog "EtapeCust1 CRASH: " & Err.Description, "ERROR"
        crashCount = crashCount + 1
        Err.Clear
        stepOK = False
    End If
    On Error GoTo 0
    If Not stepOK Then
        CustLog "ABANDON: pas de custodians", "ERROR"
        GoTo Cleanup
    End If

    ' ETAPE 2
    On Error Resume Next
    stepOK = EtapeCust2_ReadRawRisk(custNames, custPIDs)
    If Err.Number <> 0 Then
        CustLog "EtapeCust2 CRASH: " & Err.Description, "ERROR"
        crashCount = crashCount + 1
        Err.Clear
        stepOK = False
    End If
    On Error GoTo 0

    ' ETAPE 3
    If stepOK Then
        On Error Resume Next
        EtapeCust3_WriteRawResults custNames, custPIDs
        If Err.Number <> 0 Then
            CustLog "EtapeCust3 CRASH: " & Err.Description, "ERROR"
            crashCount = crashCount + 1
            Err.Clear
        End If
        On Error GoTo 0
    Else
        CustLog "EtapeCust3: SKIP", "WARN"
    End If

    ' TAUX DE CHANGE
    fxOK = False
    If stepOK Then
        On Error Resume Next
        fxOK = FetchEURRates(fxRates)
        If Err.Number <> 0 Then
            CustLog "FetchEURRates CRASH: " & Err.Description, "ERROR"
            crashCount = crashCount + 1
            Err.Clear
            fxOK = False
        End If
        On Error GoTo 0
        If Not fxOK Then
            CustLog "TAUX INDISPONIBLES: pas de conversion EUR", "ERROR"
            Set fxRates = CreateObject("Scripting.Dictionary")
            fxRates("EUR") = 1
        End If
        WriteFxDisplay fxRates
    End If

    ' ETAPE 4
    ltvOK = False
    If stepOK Then
        On Error Resume Next
        ltvOK = EtapeCust4_ReadLTV(custPIDs, ltvData)
        If Err.Number <> 0 Then
            CustLog "EtapeCust4 CRASH: " & Err.Description, "ERROR"
            crashCount = crashCount + 1
            Err.Clear
            ltvOK = False
        End If
        On Error GoTo 0
    Else
        CustLog "EtapeCust4: SKIP", "WARN"
    End If

    ' ETAPE 5
    If ltvOK Then
        On Error Resume Next
        EtapeCust5_WriteLTVResults custNames, custPIDs, ltvData, fxRates
        If Err.Number <> 0 Then
            CustLog "EtapeCust5 CRASH: " & Err.Description, "ERROR"
            crashCount = crashCount + 1
            Err.Clear
        End If
        On Error GoTo 0
    Else
        CustLog "EtapeCust5: SKIP", "WARN"
    End If

    If crashCount = 0 Then
        CustLog "========== SUCCES ==========", "OK"
    Else
        CustLog "========== " & crashCount & " CRASH(ES) ==========", "ERROR"
    End If

Cleanup:
    Application.ScreenUpdating = True
    Application.Calculation = xlCalculationAutomatic
    Application.EnableEvents = True
    elapsed = Timer - startTime
    If crashCount = 0 Then
        MsgBox "OK en " & Format(elapsed, "0.0") & "s" & vbCrLf & "Voir Log.", vbInformation
    Else
        MsgBox crashCount & " erreur(s) en " & Format(elapsed, "0.0") & "s" & vbCrLf & "Voir Log.", vbExclamation
    End If
End Sub


' ================================================================================
' TEST MACROS
' ================================================================================

Public Sub TestCust1()
    CustLogInit
    m_businessDate = ""
    Dim custNames As Object
    If EtapeCust1_ReadCustodians(custNames) Then
        MsgBox "OK: " & custNames.Count & " noms"
    Else
        MsgBox "ECHEC -- voir Log"
    End If
End Sub

Public Sub TestCust2()
    CustLogInit
    m_businessDate = ""
    Dim custNames As Object
    Dim custPIDs As Object
    If Not EtapeCust1_ReadCustodians(custNames) Then
        MsgBox "EtapeCust1 echoue": Exit Sub
    End If
    If EtapeCust2_ReadRawRisk(custNames, custPIDs) Then
        MsgBox "OK -- voir Log"
    Else
        MsgBox "ECHEC -- voir Log"
    End If
End Sub

Public Sub TestCust3()
    CustLogInit
    m_businessDate = ""
    Dim custNames As Object
    Dim custPIDs As Object
    If Not EtapeCust1_ReadCustodians(custNames) Then
        MsgBox "EtapeCust1 echoue": Exit Sub
    End If
    If Not EtapeCust2_ReadRawRisk(custNames, custPIDs) Then
        MsgBox "EtapeCust2 echoue": Exit Sub
    End If
    If EtapeCust3_WriteRawResults(custNames, custPIDs) Then
        MsgBox "OK -- voir Log et Sheet1"
    Else
        MsgBox "ECHEC -- voir Log"
    End If
End Sub

Public Sub TestCust4()
    CustLogInit
    m_businessDate = ""
    Dim custNames As Object
    Dim custPIDs As Object
    Dim ltvData As Object
    If Not EtapeCust1_ReadCustodians(custNames) Then
        MsgBox "EtapeCust1 echoue": Exit Sub
    End If
    If Not EtapeCust2_ReadRawRisk(custNames, custPIDs) Then
        MsgBox "EtapeCust2 echoue": Exit Sub
    End If
    If EtapeCust4_ReadLTV(custPIDs, ltvData) Then
        MsgBox "OK: " & ltvData.Count & " PIDs -- voir Log"
    Else
        MsgBox "ECHEC -- voir Log"
    End If
End Sub

Public Sub TestCust5()
    CustLogInit
    m_businessDate = ""
    Dim custNames As Object
    Dim custPIDs As Object
    Dim ltvData As Object
    Dim fxRates As Object
    If Not EtapeCust1_ReadCustodians(custNames) Then
        MsgBox "EtapeCust1 echoue": Exit Sub
    End If
    If Not EtapeCust2_ReadRawRisk(custNames, custPIDs) Then
        MsgBox "EtapeCust2 echoue": Exit Sub
    End If
    If Not EtapeCust4_ReadLTV(custPIDs, ltvData) Then
        MsgBox "EtapeCust4 echoue": Exit Sub
    End If
    If Not FetchEURRates(fxRates) Then
        MsgBox "Taux indisponibles", vbExclamation
        Set fxRates = CreateObject("Scripting.Dictionary")
        fxRates("EUR") = 1
    End If
    WriteFxDisplay fxRates
    If EtapeCust5_WriteLTVResults(custNames, custPIDs, ltvData, fxRates) Then
        MsgBox "OK -- voir Log et Sheet1"
    Else
        MsgBox "ECHEC -- voir Log"
    End If
End Sub
