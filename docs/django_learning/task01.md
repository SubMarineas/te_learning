@[TOC](restframework （1）：介绍与应用)

# restframework

django restframework是基于django和restful协议开发的框架，**在restful协议里，一切皆是资源，操作是通过请求方式控制**，

## restframework介绍

在开发REST API的视图中，虽然每个视图具体操作的数据不同，但增、删、改、查的实现流程基本套路化，所以这部分代码也是可以复用简化编写的：
- 增：校验请求数据 -> 执行反序列化过程 -> 保存数据库 -> 将保存的对象序列化并返回
- 删：判断要删除的数据是否存在 -> 执行数据库删除
- 改：判断要修改的数据是否存在 -> 校验请求的数据 -> 执行反序列化过程 -> 保存数据库 -> 将保存的对象序列化并返回
- 查：查询数据库 -> 将数据序列化并返回

**Django REST framework可以帮助我们简化上述两部分的代码编写，大大提高REST API的开发速度。**



## HTTP动词
对于资源的具体操作类型，由HTTP动词表示。

常用的HTTP动词有下面四个（括号里是对应的SQL命令）。

- GET（SELECT）：从服务器取出资源（一项或多项）。
- POST（CREATE）：在服务器新建一个资源。
- PUT（UPDATE）：在服务器更新资源（客户端提供改变后的完整资源）。
- DELETE（DELETE）：从服务器删除资源。

还有三个不常用的HTTP动词。

- PATCH（UPDATE）：在服务器更新(更新)资源（客户端提供改变的属性）。
- HEAD：获取资源的元数据。
- OPTIONS：获取信息，关于资源的哪些属性是客户端可以改变的。



## 状态码

> 200 OK - [GET]：服务器成功返回用户请求的数据，该操作是幂等的（Idempotent）。
> 200 OK - [GET]：服务器成功返回用户请求的数据，该操作是幂等的（Idempotent）。
201 CREATED - [POST/PUT/PATCH]：用户新建或修改数据成功。
202 Accepted - ：表示一个请求已经进入后台排队（异步任务）
204 NO CONTENT - [DELETE]：用户删除数据成功。
400 INVALID REQUEST - [POST/PUT/PATCH]：用户发出的请求有错误，服务器没有进行新建或修改数据的操作，该操作是幂等的。
401 Unauthorized - ：表示用户没有权限（令牌、用户名、密码错误）。
403 Forbidden -  表示用户得到授权（与401错误相对），但是访问是被禁止的。
404 NOT FOUND - ：用户发出的请求针对的是不存在的记录，服务器没有进行操作，该操作是幂等的。
406 Not Acceptable - [GET]：用户请求的格式不可得（比如用户请求JSON格式，但是只有XML格式）。
410 Gone -[GET]：用户请求的资源被永久删除，且不会再得到的。
422 Unprocesable entity - [POST/PUT/PATCH] 当创建一个对象时，发生一个验证错误。
500 INTERNAL SERVER ERROR - [*]：服务器发生错误，用户将无法判断发出的请求是否成功。





# CBV和FBV


**FBV（function base views）** 就是在视图里使用函数处理请求。**CBV（class base views）** 就是在视图里使用类处理请求。两种方式本身在速度与读写过程上并没有太大区别，全看哪种方便用哪种。

## 1. FBV
1.1 url.py:
```javascript
// An highlighted block
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^publisher_list/', views.publisher_list),
    url(r'^add_publisher/', views.book_list),
    ]
```

1.2 views.py:

```javascript
// An highlighted block
def publisher_list(request):
    # 去数据库查出所有的出版社,填充到HTML中,给用户返回
    ret = models.Publisher.objects.all().order_by("id")
    return render(request, "publisher_list.html", {"publisher_list": ret})
```
FBV看起来比较检验明了，下面再看CBV的方式：

## 2. CBV
2.1 url.py

```javascript
// An highlighted block
urlpatterns = [
    url(r'^admin/', admin.site.urls),
    url(r'^publishes/$', views.PublishView.as_view(),name="publish"), 
    url(r'^publishes/(?P<pk>\d+)/$', views.PublishDetailView.as_view(),name="detailpublish"),
    ]
```
2.2 views.py

```javascript
// An highlighted block
	class PublishView(APIView):
   	 	def get(self,request):
   	 		publish_list = Publish.objects.all()
        	ps = PublishModelSerializers(publish_list, many=True)
        	return Response(ps.data)
```


我们发现，虽然函数简单明了，但如果只用函数来开发，有很多面向对象的优点就错失了（继承、封装、多态）。所以Django在后来加入了Class-Based-View。可以让我们用类写View，然后通过反射执行as_view()方法，这样做的优点主要下面两种：

1. 提高了代码的复用性，可以使用面向对象的技术，比如Mixin（多继承）
2. 可以用不同的函数针对不同的HTTP方法处理，而不是通过很多if判断，提高代码可读性

下面都将以类的方式来介绍该机制

---------------------------


# django  restframework


## 原生django中的request


```javascript
def post(self, request):
    print(request.body)
    print(request.POST)
```
		  "GET url?a=1&b=2 http/1.1\r\user_agent:Google\r\ncontentType:urlencoded\r\n\r\n"
		  "POST url http/1.1\r\user_agent:Google\r\ncontentType:urlencoded\r\n\r\na=1&b=2"

上面这两段都是原生django中通过WSGI处理后保存到request中的body方法和post方法，body方法是通过获取请求体中得到的数据返回给后端，也就是GET url后的a=1&b=2，而post是返回最后的数据。


 

## restframework中的request

对于restframework的request，我们需要了解下在restframework里请求流程。

**REST framework 传入视图的request对象不再是Django默认的HttpRequest对象，而是REST framework提供的扩展了HttpRequest类的Request类的对象。**

REST framework 提供了Parser解析器，在接收到请求后会自动根据Content-Type指明的请求数据类型（如JSON、表单等）将请求数据进行parse解析，解析为类字典对象保存到Request对象中。

它和django大致相同，因为它的APIView继承是django的View，但在APiView中**重写了dispatch方法**

看到这段代码：
```javascript
url(r'^publishers/$', views.PublishViewSet.as_view(),name="publish_list"),
```
 执行PublishViewSet就是APIView的as_view方法
```javascript
class APIView(View):
```

 　　APIView继承了View，APIView中有as_view方法，所以会执行这个方法，方法中有这么一句代码
```javascript
view = super(APIView, cls).as_view(**initkwargs)
```

 　　最终还是执行了父类里的as_view方法，所以最终执行结果，得到这么这个view函数。
 　　下面我们就去源码中看看这个函数的执行顺序：
```javascript
         def view(request, *args, **kwargs):
            self = cls(**initkwargs)
            if hasattr(self, 'get') and not hasattr(self, 'head'):
                self.head = self.get
            self.request = request
            self.args = args
            self.kwargs = kwargs
            return self.dispatch(request, *args, **kwargs)
```

 　　当请求来时，会执行view函数，然后结果调用了dispatch方法，而这里dispatch方法则不是View里的，因为APIView中重写了父类中的dispatch方法，并且是整个rest_framework中最重要的部分，实现了大部分逻辑。

```javascript
 　def dispatch(self, request, *args, **kwargs):
        """
        `.dispatch()` is pretty much the same as Django's regular dispatch,
        but with extra hooks for startup, finalize, and exception handling.
        """
        self.args = args
        self.kwargs = kwargs
        request = self.initialize_request(request, *args, **kwargs)
        self.request = request
        self.headers = self.default_response_headers  # deprecate?

        try:
            self.initial(request, *args, **kwargs)

            # Get the appropriate handler method
            if request.method.lower() in self.http_method_names:
                handler = getattr(self, request.method.lower(),
                                  self.http_method_not_allowed)
            else:
                handler = self.http_method_not_allowed

            response = handler(request, *args, **kwargs)

        except Exception as exc:
            response = self.handle_exception(exc)

        self.response = self.finalize_response(request, response, *args, **kwargs)
        return self.response
        
```

所以我们可以总结，dispatch是通过view的函数的执行而被调用，那么它返回的结果就是view函数返回的结果，而view函数返回的结果就是as_view()方法返回的结果。也就是通过这样的方式获取到了请求方式并执行。



d
 [1]:https://www.cnblogs.com/liwenzhou/p/9338256.html
 [2]: http://www.ruanyifeng.com/blog/developer/
 [3]: https://www.cnblogs.com/yuanchenqi/articles/8719520.html
