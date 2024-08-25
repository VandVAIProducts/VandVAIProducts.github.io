(function () {

    Renderer = function (canvas) {
        canvas = $(canvas).get(0)
        var ctx = canvas.getContext("2d")
        var particleSystem = null

        var palette = {
            "Network": "#4D7A00",
        }

        var that = {
            init: function (system) {
                particleSystem = system
                particleSystem.screen({
                    padding: [200, 200, 200, 200],
                    step: .02
                }) // have the ‘camera’ zoom somewhat slowly as the graph unfolds 
                $(window).resize(that.resize)
                that.resize()

                that.initMouseHandling()
            },
            redraw: function () {
                if (particleSystem === null) return

                ctx.clearRect(0, 0, canvas.width, canvas.height)
                ctx.strokeStyle = "#d3d3d3"
                ctx.lineWidth = 1
                ctx.beginPath()
                particleSystem.eachEdge(function (edge, pt1, pt2) {

                    var weight = Math.max(1, edge.data.weight / 100)
                    var color = null // edge.data.color
                    if (!color || ("" + color).match(/^[ \t]*$/)) color = null

                    if (color !== undefined || weight !== undefined) {
                        ctx.save()
                        ctx.beginPath()

                        if (!isNaN(weight)) ctx.lineWidth = weight

                        if (edge.source.data.region == edge.target.data.region) {
                            ctx.strokeStyle = palette[edge.source.data.region]
                        }

                        ctx.fillStyle = null
                        ctx.moveTo(pt1.x, pt1.y)
                        ctx.lineTo(pt2.x, pt2.y)
                        ctx.stroke()
                        ctx.restore()
                    } else {
                        ctx.moveTo(pt1.x, pt1.y)
                        ctx.lineTo(pt2.x, pt2.y)
                    }
                })
                ctx.stroke()

                particleSystem.eachNode(function (node, pt) {
                    // drawing a text label (awful alignment jitter otherwise...)
                    var w = ctx.measureText(node.data.label || "").width + 6
                    var h = 14
                    var label = node.data.label
                    if (!(label || "").match(/^[ \t]*$/)) {
                        pt.x = Math.floor(pt.x)
                        pt.y = Math.floor(pt.y)
                    } else {
                        label = null
                    }

                    ctx.clearRect(pt.x - w / 2, pt.y - h / 2, w, h)

                    // draw the text
                    if (label) {
                        ctx.font = "16px Helvetica";
                        ctx.textAlign = "center"
                        ctx.fillStyle = node.data.color                        
                        ctx.fillText(label || "", pt.x, pt.y + 4)
                    }
                })
            },

            resize: function () {
                var w = $(window).width(),
                    h = $(window).height();
                canvas.width = w; canvas.height = h // resize the canvas element to fill the screen
                particleSystem.screenSize(w, h) // inform the system so it can map coords for us
                that.redraw()
            },

            initMouseHandling: function () {
                selected = null;
                nearest = null;
                var dragged = null;

                $(canvas).mousedown(function (e) {
                    var pos = $(this).offset();
                    var p = { x: e.pageX - pos.left, y: e.pageY - pos.top }
                    selected = nearest = dragged = particleSystem.nearest(p);

                    if (selected.node !== null) {
                        dragged.node.tempMass = 50
                        dragged.node.fixed = true
                    }
                    return false
                });

                $(canvas).mousemove(function (e) {
                    var pos = $(this).offset();
                    var s = { x: e.pageX - pos.left, y: e.pageY - pos.top };

                    nearest = particleSystem.nearest(s);
                    if (!nearest) return
                    if (dragged !== null && dragged.node !== null) {
                        var p = particleSystem.fromScreen(s)
                        dragged.node.p = { x: p.x, y: p.y }
                    }
                    return false
                });

                $(window).bind('mouseup', function (e) {
                    if (dragged === null || dragged.node === undefined) return
                    dragged.node.fixed = false
                    dragged.node.tempMass = 100
                    dragged = null;
                    selected = null
                    return false
                });
            },
        }
        return that
    }

    var Maps = function (elt) {
        var sys = arbor.ParticleSystem(4000, 500, 0.5, 55)
        sys.renderer = Renderer("#viewport")
        var dom = $(elt)
        var _links = dom.find('ul')
        var _sources = {
            notes: 'Build using <a target="_blank" href="http://arborjs.org/">ArborJS</a>',
        }

        var _maps = {
            network: { title: "Network", p: { stiffness: 500 }, source: _sources.notes },
        }

        var that = {
            init: function () {
                $.each(_maps, function (stub, map) {
                    _links.append("<li><a href='#/" + stub + "' class='" + stub + "'>" + map.title + "</a></li>")
                })
                _links.find('li > a').click(that.mapClick)
                _links.find('.network').click()
                return that
            },
            mapClick: function (e) {
                var selected = $(e.target)
                var newMap = selected.attr('class')
                if (newMap in _maps) that.selectMap(newMap)
                _links.find('li > a').removeClass('active')
                selected.addClass('active')
                return false
            },
            selectMap: function (map_id) {
                $.getJSON("assets/json/" + map_id + ".json", function (data) {
                    // load the raw data into the particle system as is (since it's already formatted correctly for .merge)
                    var nodes = data.nodes
                    $.each(nodes, function (name, info) {
                        info.label = name.replace(/(people's )?republic of /i, '').replace(/ and /g, ' & ')
                    })
                    sys.merge({ nodes: nodes, edges: data.edges })
                    sys.parameters(_maps[map_id].p)
                    $("#dataset").html(_maps[map_id].source)
                })
            }
        }
        return that.init()
    }

    $(document).ready(function () {
        var mcp = Maps("#maps")
    })
})()