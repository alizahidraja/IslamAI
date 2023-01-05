from flask import Flask , render_template , request , redirect , url_for

app=Flask(__name__)

@app.route("/",methods=["POST","GET"])
def home():
    if request.method=="POST":
        QUERy=request.form["query"]
        return redirect(url_for("query_results",Query=QUERy))
    else:
        return render_template("main.html")

@app.route("/search<Query>")
def query_results(Query):
    return render_template("query_results.html",query=Query)

if __name__=="__main__":
    app.run(debug=True)