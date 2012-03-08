<html>
<head>
<title>Sensitivity analysis of electrophysiology model</title>
</head>
<body>
%if "clear" in query:
%   for k in "title", "parameters", "clear":
%       query.pop(k, None)
%   end
%end
{{query.keys()}}
{{query.parameters}}
<h1>Sensitivity analysis of electrophysiology models</h1>
<form action="{{path}}">
<select name=title size=10 style="width: 50%">
%for title in workspaces.title:
    %selected = "selected=True" if (title == query.title) else ""
    <option value="{{title}}" {{selected}}>{{title}}</option>
%end
</select>
%if model:
    <select multiple name=parameters size={{len(model.dtype.p)}}>
    %for k in model.dtype.p.names:
        %selected = "selected=True" if (k in query.parameters) else ""
        <option value={{k}} {{selected}}>{{k}}</option>
    %end
    </select>
%end
<input type=submit />
<input type=submit name=clear value="Clear" />
</form>
</body>
</html>