# Troubleshooting Guide - Photon Neuromorphics SDK

## Common Issues

### Import Errors
**Symptom**: `ModuleNotFoundError: No module named 'photon_neuro'`

**Solutions**:
1. Verify installation: `pip list | grep photon`
2. Check Python path: `python -c "import sys; print(sys.path)"`
3. Reinstall package: `pip install --force-reinstall photon-neuromorphics`

### Memory Issues
**Symptom**: High memory usage, OOM kills

**Solutions**:
1. Monitor memory usage: `docker stats`
2. Increase container memory limits
3. Check for memory leaks in application code
4. Enable garbage collection: `import gc; gc.collect()`

### Performance Issues
**Symptom**: Slow response times, high latency

**Solutions**:
1. Check resource utilization: `htop`, `iostat`
2. Profile application: `python -m cProfile`
3. Enable caching: Set `ENABLE_CACHE=true`
4. Scale horizontally: Add more replicas

### Container Issues
**Symptom**: Container fails to start

**Solutions**:
1. Check container logs: `docker logs <container_id>`
2. Verify image build: `docker build --no-cache`
3. Check resource limits and requests
4. Validate configuration files

### Network Issues
**Symptom**: Service unreachable

**Solutions**:
1. Check service status: `kubectl get svc`
2. Verify endpoints: `kubectl get endpoints`
3. Test network connectivity: `telnet <service_ip> <port>`
4. Check firewall rules and network policies

## Debugging Commands

### Docker Debugging
```bash
# Inspect container
docker inspect <container_id>

# Execute into container
docker exec -it <container_id> /bin/bash

# Check resource usage
docker stats

# View container logs
docker logs -f <container_id>
```

### Kubernetes Debugging
```bash
# Describe resources
kubectl describe pod <pod_name> -n photon-neuro
kubectl describe deployment photon-neuro-app -n photon-neuro

# Check events
kubectl get events -n photon-neuro --sort-by=.metadata.creationTimestamp

# Port forwarding for debugging
kubectl port-forward pod/<pod_name> 8080:8080 -n photon-neuro

# Execute into pod
kubectl exec -it <pod_name> -n photon-neuro -- /bin/bash
```

### Application Debugging
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Check application health
curl http://localhost:8080/health

# View application metrics
curl http://localhost:8080/metrics

# Test import functionality
python -c "import photon_neuro; print('OK')"
```

## Log Analysis

### Log Locations
- Container logs: `docker logs <container>`
- Kubernetes logs: `kubectl logs <pod>`
- Application logs: `/app/logs/` (if volume mounted)

### Common Log Patterns
```bash
# Error patterns
grep "ERROR" /var/log/photon-neuro.log

# Memory warnings
grep "Memory" /var/log/photon-neuro.log

# Performance issues
grep "slow" /var/log/photon-neuro.log
```

## Performance Profiling

### Memory Profiling
```python
import tracemalloc
tracemalloc.start()

# Your code here

current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
tracemalloc.stop()
```

### CPU Profiling
```bash
# Profile application
python -m cProfile -o profile.stats your_script.py

# Analyze results
python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(10)"
```

## Getting Help

### Internal Resources
- Check monitoring dashboards
- Review application logs
- Consult architecture documentation

### External Resources
- GitHub Issues: Report bugs and feature requests
- Documentation: https://photon-neuro.io
- Community: Stack Overflow (tag: photon-neuromorphics)

### Emergency Contacts
- On-call engineer: [Contact information]
- System administrator: [Contact information]
- Product owner: [Contact information]
