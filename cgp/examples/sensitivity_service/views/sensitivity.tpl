<html>
<head>
<title>Sensitivity analysis of electrophysiology model</title>
</head>
<body>
<h1>Sensitivity analysis of electrophysiology models</h1>
<form action="">
<select name="model" size=10 style="width: 50%">
%for title in model.title:
    <option value="{{title}}">{{title}}</option>
%end
</select>
</form>
</body>
</html>