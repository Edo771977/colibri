param([string]$Path)
$err = $null; $tok = $null
$ast = [System.Management.Automation.Language.Parser]::ParseFile($Path, [ref]$tok, [ref]$err)
if ($err -and $err.Count) {
    "PARSE: $($err.Count) errori"
    $err | ForEach-Object { "  riga {0}: {1}" -f $_.Extent.StartLineNumber, $_.Message }
    exit 1
}
"PARSE: ok"
# funzioni definite piu' di una volta -- il difetto che ha rotto la misura
$fn = $ast.FindAll({ $args[0] -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)
$dup = $fn | Group-Object Name | Where-Object Count -gt 1
if ($dup) { $dup | ForEach-Object { "DUPLICATA: {0} definita {1} volte" -f $_.Name, $_.Count }; exit 1 }
"funzioni: {0}, nessuna duplicata" -f $fn.Count
# variabili lette e mai assegnate in nessuno scope del file
$assigned = @($ast.FindAll({ $args[0] -is [System.Management.Automation.Language.AssignmentStatementAst] }, $true) |
    ForEach-Object { $_.Left } | Where-Object { $_ -is [System.Management.Automation.Language.VariableExpressionAst] } |
    ForEach-Object { $_.VariablePath.UserPath })
$params = @($ast.FindAll({ $args[0] -is [System.Management.Automation.Language.ParameterAst] }, $true) |
    ForEach-Object { $_.Name.VariablePath.UserPath })
# le variabili di foreach sono assegnate dal ciclo, non da un AssignmentStatement
$loops = @($ast.FindAll({ $args[0] -is [System.Management.Automation.Language.ForEachStatementAst] }, $true) |
    ForEach-Object { $_.Variable.VariablePath.UserPath })
$auto = @('_','args','PSScriptRoot','PSItem','Matches','LASTEXITCODE','ErrorActionPreference','env','null','true','false','PSVersionTable','PWD')
$used = @($ast.FindAll({ $args[0] -is [System.Management.Automation.Language.VariableExpressionAst] }, $true) |
    ForEach-Object { $_.VariablePath.UserPath }) | Sort-Object -Unique
$unknown = $used | Where-Object { $_ -notin $assigned -and $_ -notin $params -and $_ -notin $loops -and $_ -notin $auto -and $_ -notlike 'env:*' }
if ($unknown) { "MAI ASSEGNATE: {0}" -f ($unknown -join ', '); exit 1 }
"variabili: nessuna lettura di variabile mai assegnata"
# I nomi di variabile in PowerShell NON distinguono maiuscole e minuscole:
# $B e $b sono la stessa variabile. Due nomi che differiscono solo per il
# caso sono quasi sempre due variabili DIVERSE nell'intenzione di chi scrive,
# e una sovrascrive l'altra in silenzio.
# -CaseSensitive: senza, Sort-Object -Unique e' esso stesso insensibile al
# caso e collassa $A e $a prima che il controllo possa vederli.
$all = @($assigned + $params + $loops) | Sort-Object -Unique -CaseSensitive
$collide = $all | Group-Object { $_.ToLowerInvariant() } | Where-Object Count -gt 1
if ($collide) {
    $collide | ForEach-Object { "COLLISIONE DI CASO: {0} -- in PowerShell sono la STESSA variabile" -f ($_.Group -join ' / ') }
    exit 1
}
"nomi: nessuna collisione fra maiuscole e minuscole"
