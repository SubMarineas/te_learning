@[TOC](restframework （3）：视图应用与源码解析)





# 总体概括

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181028201530753.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3N1Ym1hcmluZWFz,size_27,color_FFFFFF,t_70)

这篇博文主要将围绕上面这张思维导图进行，下面我们将从应用讲到源码解析。


## Serializers.py:

我们此处全部用ModelSerializers替代，第一是因为方便，第二就是见效快，并且本篇博客的重点并不在这里，还有就是上一张数据表在我做JWT验证的时候被我误删了。。。所以，so，表结构和上一篇差不太多，序列化器部分如下：

```javascript
class PublishModelSerializers(serializers.ModelSerializer):
    class Meta:
        model=Publish
        fields="__all__"


class BookModelSerializers(serializers.ModelSerializer):
    class Meta:
        model = Book
        fields = "__all__"


class AuthorModelSerializers(serializers.ModelSerializer):
    class Meta:
        model = Author
        fields = "__all__"
```


# 视图类运用
-----------------
## 第一种表示方法：

在上一篇中，我们快速实例了Book和Author两张表的数据序列化，但对比发现，代码高度重复，显得有些冗余，下面我们再回过头来看看：

```javascript
class BookView(APIView):

    def get(self,request):
        print("request.user",request.user)
        publish_list=Publish.objects.all()
        bs=PublishModelSerializers(book_list,many=True)
        return Response(bs.data)

    def post(self,request):
        # post请求的数据
        bs=PublishModelSerializers(data=request.data)
        if bs.is_valid():
            print(bs.validated_data)
            bs.save()# create方法
            return Response(bs.data)
        else:
            return Response(bs.errors)


class PublishDetailView(APIView):

    def get(self,request,id):
        publish=Publish.objects.filter(pk=id).first()
        bs=PublishModelSerializers(publish)
        return Response(bs.data)

    def put(self,request,id):
        publish=Publish.objects.filter(pk=id).first()
        bs=PublishModelSerializers(publish,data=request.data)
        if bs.is_valid():
            bs.save()
            return Response(bs.data)
        else:
            return Response(bs.errors)

    def delete(self,request,id):
        Publish.objects.filter(pk=id).delete()
        return Response()
```
洋洋洒洒38行，多一行受累，少一行受罪，这是我们上一篇基本原封未动的代码，其实基本都是可以封装的，也正好restframework提供了这种便利。下面真正进入今天的正题。


## 第二种表示：Mixins模块

```javascript
from rest_framework import mixins
from rest_framework import generics

class BookViewSet(mixins.ListModelMixin,
                  mixins.CreateModelMixin,
                  generics.GenericAPIView):

    queryset = Book.objects.all()
    serializer_class = BookSerializers

    def get(self, request, *args, **kwargs):
        return self.list(request, *args, **kwargs)

    def post(self, request, *args, **kwargs):
        return self.create(request, *args, **kwargs)



class BookDetailViewSet(mixins.RetrieveModelMixin,
                    mixins.UpdateModelMixin,
                    mixins.DestroyModelMixin,
                    generics.GenericAPIView):
    queryset = Book.objects.all()	# 固定字段，
    serializer_class = BookSerializers		# 固定字段

    def get(self, request, *args, **kwargs):
        return sef.retrieve(request, *args, **kwargs)

    def put(self, request, *args, **kwargs):
        return self.update(request, *args, **kwargs)

    def delete(self, request, *args, **kwargs):
        return self.destroy(request, *args, **kwargs)
```

这里我们用到了多继承的方法，当执行过程中，会按照继承的顺序去调用相应的逻辑处理部分，我们会发现这种方式比上面简洁了很多，少了逻辑处理部分，多了两个固定参数，这两个是指定去哪张表里查，不能用别的变量名代替，另外就是类中的方法名和返回值的有些出入，这个我们再后面讲源码的时候再说，其实是通过反射的原理将请求换了种名字。

然后我们利用postman发送get请求，状态码为200，可以看到我们表的json格式数据为：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20181028201736345.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3N1Ym1hcmluZWFz,size_27,color_FFFFFF,t_70)


## 第三种方式：通用类generics

不多讲，附上代码：

```javascript
from rest_framework import generics

class AuthorView(generics.ListCreateAPIView):
    queryset=Author.objects.all()
    serializer_class =AuthorModelSerializers

class AuthorDetailView(generics.RetrieveUpdateDestroyAPIView):
    queryset = Author.objects.all()
    serializer_class = AuthorModelSerializers
```

这里我们发现，全部的继承关系，都是基于generics，那么也就是说它里面封装好了各种单一或者组合式的APIView可以让我们调用，也就相当于替换了mixin模块的功能，然后如果细心的读者会发现既然包含了所有的mixin模块功能，为什么最前面的那张流程图没有画？其实并不是我不画，只是它里面的类有点多，还有各种组合感觉影响美观。。。所以在这里提一下。

> 1） CreateAPIView
提供 post 方法
继承自： GenericAPIView、CreateModelMixin
>2）ListAPIView
提供 get 方法
继承自：GenericAPIView、ListModelMixin
3）RetireveAPIView
提供 get 方法
继承自: GenericAPIView、RetrieveModelMixin
4）DestoryAPIView
提供 delete 方法
继承自：GenericAPIView、DestoryModelMixin
5）UpdateAPIView
提供 put 和 patch 方法
继承自：GenericAPIView、UpdateModelMixin
6）RetrieveUpdateAPIView
提供 get、put、patch方法
继承自： GenericAPIView、RetrieveModelMixin、UpdateModelMixin
7）RetrieveUpdateDestoryAPIView
提供 get、put、patch、delete方法
继承自：GenericAPIView、RetrieveModelMixin、UpdateModelMixin、DestoryModelMixin


最后，我想看看第二本书《长安乱》是谁写的，那么我们用精确查找的方式，见结果(推荐可以看看，还有另两本书，不过另外两个就是系列作了)：


![在这里插入图片描述](https://img-blog.csdnimg.cn/20181028203029186.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3N1Ym1hcmluZWFz,size_27,color_FFFFFF,t_70)

## 第四种方式：终极方案

url.py：
```javascript
    url(r'^authors/$', views.AuthorModelView.as_view({"get":"list","post":"create"}),name="author"),
    url(r'^authors/(?P<pk>\d+)/$', views.AuthorModelView.as_view({"get":"retrieve","put":"update","delete":"destroy"}),name="detailauthor"),
```

views.py:
```javascript
from rest_framework.viewsets import  ModelViewSet

class BookViewSet(viewsets.ModelViewSet):
    queryset = Book.objects.all()
    serializer_class = BookSerializers
```

视图部分就四行代码，然后就可以实现上面我们写的一张表中的所有功能了，和第一种相比少了太多太多，这里就不再测试了，因为即使测试其实还是不会懂为什么短短四行能搞定这个，那么我们下面就进入源码解析。



# 视图源码解析


![在这里插入图片描述](https://img-blog.csdnimg.cn/20181028204537753.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3N1Ym1hcmluZWFz,size_27,color_FFFFFF,t_70)


我这里又重画了一下思维导图，加入了源码部分，下面就一个个来介绍：

## Mixins的五个类：
首先点开ModelViewSet类：
```javascript
class ModelViewSet(mixins.CreateModelMixin,
                   mixins.RetrieveModelMixin,
                   mixins.UpdateModelMixin,
                   mixins.DestroyModelMixin,
                   mixins.ListModelMixin,
                   GenericViewSet):
```
点进第一个类中：

### CreateModelMixin：

```javascript
class CreateModelMixin(object):
    """
    Create a model instance.
    """
    def create(self, request, *args, **kwargs):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        self.perform_create(serializer)
        headers = self.get_success_headers(serializer.data)
        return Response(serializer.data, status=status.HTTP_201_CREATED, headers=headers)

    def perform_create(self, serializer):
        serializer.save()

    def get_success_headers(self, data):
        try:
            return {'Location': str(data[api_settings.URL_FIELD_NAME])}
        except (TypeError, KeyError):
            return {}
```
这是CreateModelMixin类下的源码，首先通过request获取请求得到的信息传给参数serializer，对它进行用户Token验证，如果用户通过，将其保存并且通过return {'Location': str(data[api_settings.URL_FIELD_NAME])}保存一个临时头部数据，最后将序列化的数据，响应状态码（201 Created:请求已经被实现）、头部一并返回即可。
另外补充，get_serializer方法是继承自GenericAPIView类的序列化数据。


### RetrieveModelMixin：

```javascript
class RetrieveModelMixin(object):
    """
    Retrieve a model instance.
    """
    def retrieve(self, request, *args, **kwargs):
        instance = self.get_object()
        serializer = self.get_serializer(instance)
        return Response(serializer.data)
```

这是RetrieveModelMixin下的源码，首先同样get_object和get_serializer是继承自GenericAPIView类中的方法，这里需要注意一下get_object的这条语句：
```javascript
lookup_url_kwarg = self.lookup_url_kwarg or self.lookup_field
```
因为这个函数后面部分都是对lookup_url_kwarg这个参数进行处理，而给这个参数赋值的两个，前者在最前面的初始参数是None，所以我们又跳到后者lookup_field，发现它的默认值是"pk"，所以这也是为什么我们之前只要是精确查找路由部分正则后的参数都设为pk的原因。

```javascript
    def get_object(self):
        """
        Returns the object the view is displaying.

        You may want to override this if you need to provide non-standard
        queryset lookups.  Eg if objects are referenced using multiple
        keyword arguments in the url conf.
        """
        queryset = self.filter_queryset(self.get_queryset())

        # Perform the lookup filtering.
        lookup_url_kwarg = self.lookup_url_kwarg or self.lookup_field

        assert lookup_url_kwarg in self.kwargs, (
            'Expected view %s to be called with a URL keyword argument '
            'named "%s". Fix your URL conf, or set the `.lookup_field` '
            'attribute on the view correctly.' %
            (self.__class__.__name__, lookup_url_kwarg)
        )

        filter_kwargs = {self.lookup_field: self.kwargs[lookup_url_kwarg]}
        obj = get_object_or_404(queryset, **filter_kwargs)

        # May raise a permission denied
        self.check_object_permissions(self.request, obj)

        return obj
```

总结：返回详情视图所需的模型类数据对象，默认使用lookup_field参数来过滤queryset。 在试图中可以调用该方法获取详情信息的模型类对象。若详情访问的模型类对象不存在，会返回404。该方法会默认使用APIView提供的check_object_permissions方法检查当前对象是否有权限被访问。
1. lookup_field 查询单一数据库对象时使用的条件字段，默认为'pk'
2. lookup_url_kwarg 查询单一数据时URL中的参数关键字名称，默认与look_field相同


### UpdateModelMixin


```javascript
class UpdateModelMixin(object):
    """
    Update a model instance.
    """
    def update(self, request, *args, **kwargs):
        partial = kwargs.pop('partial', False)
        instance = self.get_object()
        serializer = self.get_serializer(instance, data=request.data, partial=partial)
        serializer.is_valid(raise_exception=True)
        self.perform_update(serializer)

        if getattr(instance, '_prefetched_objects_cache', None):
            # If 'prefetch_related' has been applied to a queryset, we need to
            # forcibly invalidate the prefetch cache on the instance.
            instance._prefetched_objects_cache = {}

        return Response(serializer.data)
```
和前面的create类似，意思通过instance = self.get_object()获取要修改的数据，然后get_serializer拿到序列化器、进行请求数据合法校验、然后通过Token进行用户验证、然后将更新好的数据返回return Response(serializer.data)

### DestroyModelMixin

```javascript
class DestroyModelMixin(object):
    """
    Destroy a model instance.
    """
    def destroy(self, request, *args, **kwargs):
        instance = self.get_object()
        self.perform_destroy(instance)
        return Response(status=status.HTTP_204_NO_CONTENT)

    def perform_destroy(self, instance):
        instance.delete()
```
通过pk(或者其他标识)获取要删除的数据，然后instance.delete()删除，最后返回return Response(status=status.HTTP_204_NO_CONTENT) 
HTTP 204(no content)表示响应执行成功,但没有数据返回,浏览器不用刷新。这个除了调用get_object方法获取数据，其它都在类范围内。

### ListModelMixin

```javascript
class ListModelMixin(object):
    """
    List a queryset.
    """
    def list(self, request, *args, **kwargs):
        queryset = self.filter_queryset(self.get_queryset())

        page = self.paginate_queryset(queryset)
        if page is not None:
            serializer = self.get_serializer(page, many=True)
            return self.get_paginated_response(serializer.data)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)
```


---------------------



queryset = self.filter_queryset(self.get_queryset())，ListModelMixin类list方法获取了这个queryset，同样该方法是在GenericAPIView类中，然后在判断是否分页，最后获取self.get_serializer(queryset, many=True)序列化类，最后将序列化后的结果serializer.data作为返回return Response(serializer.data)返回即可。这里就不再分析中间这段分页的含义了，分页也算一个组件，如果以后有时间，可以再开一贴分析一下，我们只要看懂前面和后面就行了。


## 五个类小结

其实我们发现上面这五个类进行的功能并不多，只是说帮我们省略了逻辑处理部分，还有解决了一些零零碎碎的小问题，那么我们定义的路由，以及视图部分定义的类，怎样才能让django识别到，并且既然是CBV，那么就少不了as_view()方法，那么它在哪，下面就让我们进入GenericViewSet下。

## GenericViewSet

```javascript
class GenericViewSet(ViewSetMixin, generics.GenericAPIView):
    """
    The GenericViewSet class does not provide any actions by default,
    but does include the base set of generic view behavior, such as
    the `get_object` and `get_queryset` methods.
    """
    pass
```
直接进入继承类。

<br >

### ViewSetMixin
我们先进入ViewSetMixin模块类中，发现它类下的第一个便是as_view方法，然后再进一步可以看到view函数，我们此处需要的就是它：

```javascript
def view(request, *args, **kwargs):
    self = cls(**initkwargs)
    # We also store the mapping of request methods to actions,
    # so that we can later set the action attribute.
    # eg. `self.action = 'list'` on an incoming GET request.
    self.action_map = actions

    # Bind methods to actions
    # This is the bit that's different to a standard view
    for method, action in actions.items():
        handler = getattr(self, action)
        setattr(self, method, handler)

    if hasattr(self, 'get') and not hasattr(self, 'head'):
        self.head = self.get

    self.request = request
    self.args = args
    self.kwargs = kwargs

    # And continue as usual
    return self.dispatch(request, *args, **kwargs)
```



```javascript
    for method, action in actions.items():
        handler = getattr(self, action)
        setattr(self, method, handler)
```
这一段的意思是遍历actions里的数据，method接收的是字典的键，而action接收的是值，actions就是我们在url传递参数，然后通过getattr反射将action的值给handler，最后执行setattr，这个方法的意思是给对象的属性赋值，若属性不存在，先创建再赋值。也就是说将method方法里的替换成handler。

### APIView

之后找到我们想要定义的dispatch方法的回调函数里，这个是在APIView模块类中，并且算是重写了dispatch方法，如果还想刨根揭底的话，就会发现它后面还有一个dispatch方法，是View模块下的，这个可以看上面我画的流程图。所以dispatch方法的返回值，就是view的返回值，view的返回值就是as_view的返回值，那么我们整个源码逻辑就都通了。
证毕！



# 总结

发现又写了这么多，和上一篇差不多，其实我感觉这两篇都能分开来算两份，那么这样一来就能水四篇。。！那感觉这个月底就能争取一个分类里写五篇，拿到徽章了。不过初来乍到，要认真点好，以后看情况水。`･ω･′
绕回正题，这篇博文写了一天，总体来讲节奏要比上一篇序列化快，可能也是视图理解得更好吧，大概的都在前面写清楚了，下一篇restframework应该是认证组件分析，希望思路还能和视图一样清晰。


















1. https://blog.csdn.net/u013210620/article/details/79879611
2. http://www.cnblogs.com/yangxt90/articles/8746825.html
3. http://www.cnblogs.com/cenyu/p/5713686.html
4. https://www.cnblogs.com/liwenzhou/p/9398959.html
