package com.nju.util;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

public class ParserUtil {
    /**
     * 生成新形式的方法签名字符串，形式为MethodName(paramTypes)->receiveType
     * @param methodSignature
     * @return
     */
    public static String generateMethodSignature(String methodSignature, String methodName) {
        int pos = methodSignature.indexOf(methodName);
        String newSignature = methodSignature.substring(pos).trim() + "->" +
                methodSignature.substring(0, pos).trim();
        return newSignature;
    }

    /**
     * 从获取的javadoc描述字符串中获取第一句作为整体简短描述
     * @param description
     * @return
     */
    public static String generateBriefDescription(String description) {
        String strDocDescription = description.replaceAll("(\\r\\n)", " ");
        int endPos = strDocDescription.indexOf(". ");
        return endPos == -1 ? strDocDescription : strDocDescription.substring(0, endPos);
    }

    /**
     * 去除带有<>的标签
     * @param description
     * @return
     */
    public static String removeTag(String description) {
        description = description.replaceAll(
                "(<code>|</code>|<a[^>]*>|</a>|<i>|</i>|</sup>|<em>|</em>|<cite>|</cite>|<p>|</p>|<b>|</b>|<I>|</I>|<pre>|</pre>|<strong>|</strong>|<blockquote>|</blockquote>)",
                "");
        description = description.replace("<sup>", "^");
        return description;
    }

    /**
     * 去除带有{\@}形式的标签
     * @param description
     * @return
     */
    public static String removeAtTag(String description) {
        String pattern = "\\{@code ((.)*?)}";
        Matcher matcher = Pattern.compile(pattern).matcher(description);
        StringBuffer sBuffer = new StringBuffer();
        while (matcher.find()) {
            matcher.appendReplacement(sBuffer, matcher.group(1));
        }
        matcher.appendTail(sBuffer);
        return sBuffer.toString();
    }
}
